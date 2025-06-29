import { RedisVectorStore } from '@langchain/redis';
import { redisVectorStore } from './redis-store';
import { PromptTemplate } from '@langchain/core/prompts';
import { HuggingFaceInference } from '@langchain/community/llms/hf'; // Caminho mais provável para HuggingFaceInference
import dotenv from 'dotenv';
import readline from 'node:readline/promises';

dotenv.config();

// --- Template de Prompt para o LLM ---
// Este prompt será usado para instruir o Mistral-7B-Instruct a gerar a resposta.
const LLM_PROMPT = PromptTemplate.fromTemplate(`[INST]
Você é um assistente de IA focado em fornecer respostas precisas, concisas e úteis com base EXCLUSIVAMENTE no contexto fornecido.
Não utilize conhecimento externo.
Se a informação necessária para responder à pergunta não estiver explicitamente no contexto, responda clara e diretamente que "Não foi possível encontrar informações relevantes para a sua consulta na nossa base de dados."

Contexto:
{context}

Pergunta: {query}

Com base no contexto fornecido, responda à pergunta de forma concisa e coerente.
[/INST]`);

// Configuração do prompt para o "Painel de Análise" (mantido para debugging/visualização)
const resultsPrompt = PromptTemplate.fromTemplate(`
 ## Análise de Documentos

 ### Contexto da Busca:
 - Termo pesquisado: "{query}"
 - Número de resultados (brutos): {rawCount}
 - Modelo de Embedding utilizado: {embeddingModel}

 ### Resultados Encontrados (após filtro de relevância):
 {results}

 ### Detalhes da Análise (para o desenvolvedor/monitoramento):
 1. Identifique os 3 principais insights
 2. Destaque termos-chave relevantes
 3. Avalie a consistência entre os documentos
`);

async function semanticSearch(query: string, k: number = 3) {
    let llmResponse = ""; // Variável para armazenar a resposta do LLM
    try {
        console.log(`\n🔍 Iniciando busca por: "${query}"`);

        if (!redisVectorStore.redisClient.isOpen) {
            await redisVectorStore.redisClient.connect();
        }

        let results: [any, number][] = []; // Definindo o tipo para 'results'
        let embeddingModelUsed = process.env.EMBEDDINGS_MODEL || 'API Padrão';

        try {
            results = await redisVectorStore.similaritySearchWithScore(query, k);
        } catch (apiError) {
            console.warn('⚠️ Alternando para embeddings locais...');
            const { HuggingFaceTransformersEmbeddings } = await import('@langchain/community/embeddings/hf_transformers');
            embeddingModelUsed = 'Xenova/all-MiniLM-L6-v2 (local)';

            const localEmbeddings = new HuggingFaceTransformersEmbeddings({
                modelName: embeddingModelUsed
            });

            const localStore = new RedisVectorStore(localEmbeddings, {
                redisClient: redisVectorStore.redisClient,
                indexName: 'artigos-embeddings'
            });

            results = await localStore.similaritySearchWithScore(query, k);
        }

        // --- Lógica de Filtro de Relevância ---
        // ATENÇÃO: Ajuste este valor (0.7) conforme seus testes.
        // Scores mais ALTOS indicam mais relevância para similaridade coseno.
        const RELEVANCE_THRESHOLD = 0.7;
        const filteredResults = results.filter(([, score]) => score >= RELEVANCE_THRESHOLD); // Alterado para `>=`

        let formattedRawResultsForDebug = ''; // Para o painel de análise
        let contextForLLM = ''; // Conteúdo que será enviado ao LLM

        if (filteredResults.length === 0) {
            formattedRawResultsForDebug = 'Nenhum documento relevante encontrado após filtragem.';
            llmResponse = "Não foi possível encontrar informações relevantes para a sua consulta na nossa base de dados.";
        } else {
            // Formatação dos resultados para o painel de análise
            formattedRawResultsForDebug = filteredResults.map(([doc, score], index) => (
                `\n### Documento ${index + 1} (Score: ${score.toFixed(3)})\n` +
                `**Título:** ${doc.metadata?.titulo || 'Sem título'}\n` +
                `**Trecho:** ${doc.pageContent.substring(0, 300)}${doc.pageContent.length > 300 ? '...' : ''}` // Maior trecho para o painel
            )).join('\n');

            // --- Prepara o Contexto para o LLM ---
            // É importante passar apenas o conteúdo relevante para o LLM
            contextForLLM = filteredResults.map(([doc, score], index) => (
                `Documento ${index + 1}:\n${doc.pageContent}`
            )).join('\n\n');

            // --- NOVO: Chamada ao LLM para Gerar a Resposta ---
            const chatModelName = process.env.CHAT_MODEL;
            // Garante que a API key esteja disponível
            const huggingFaceApiKey = process.env.HUGGINGFACEHUB_API_TOKEN; 

            if (!chatModelName || !huggingFaceApiKey) {
                console.error("Variáveis de ambiente CHAT_MODEL ou HUGGINGFACEHUB_API_TOKEN não configuradas. Não será possível gerar respostas.");
                llmResponse = "Erro: Modelo de chat ou chave API não configurados para gerar respostas.";
            } else {
                const huggingFaceLLM = new HuggingFaceInference({
                    model: chatModelName,
                    apiKey: huggingFaceApiKey, // <--- AQUI ESTÁ A SOLUÇÃO 2
                    maxTokens: 300, 
                    temperature: 0.3, 
                });

                try {
                    const llmPromptFormatted = await LLM_PROMPT.format({
                        query: query,
                        context: contextForLLM
                    });

                    const generatedText = await huggingFaceLLM.invoke(llmPromptFormatted, {
                        maxTokens: 300, 
                        temperature: 0.3, 
                    });

                    // Mistral-7B-Instruct geralmente retorna a resposta após o [/INST]
                    const cleanGeneratedText = generatedText.split('[/INST]').pop()?.trim() || generatedText.trim();
                    llmResponse = cleanGeneratedText;

                } catch (llmError: any) {
                    console.error("❌ Erro ao chamar o LLM:", llmError.message || llmError);
                    llmResponse = "Houve um erro técnico ao gerar a resposta. Por favor, tente novamente mais tarde.";
                }
            }
        }

        // --- Saída para o Usuário (Resposta Gerada) ---
        console.log('\n✅ Resposta Gerada:');
        console.log(llmResponse);

        // --- Painel de Análise (para debugging e detalhes internos) ---
        console.log('\n--- Painel de Análise (Detalhes Internos) ---');
        const analysisPrompt = await resultsPrompt.format({
            query,
            rawCount: results.length, // Mostra o número de resultados antes do filtro
            embeddingModel: embeddingModelUsed,
            results: formattedRawResultsForDebug
        });
        console.log(analysisPrompt);

        // Detalhes técnicos
        console.log('\n🔍 Detalhes Técnicos Finais:');
        console.log(`Modelo LLM: ${process.env.CHAT_MODEL || 'Não configurado'}`);
        console.log(`Modelo Embedding: ${embeddingModelUsed}`);
        console.log(`Tempo: ${new Date().toLocaleTimeString()}`);

    } finally {
        if (redisVectorStore.redisClient.isOpen) {
            await redisVectorStore.redisClient.disconnect();
        }
    }
}

// Interface interativa melhorada
async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    try {
        console.log('\n=== Sistema de Análise Semântica ===');
        console.log('📌 Modo:', process.env.HUGGINGFACEHUB_API_TOKEN ? 'API' : 'Local');

        while (true) {
            const query = await rl.question('\n🔎 Digite sua consulta (ou "sair"): ');
            if (query.toLowerCase() === 'sair') break;

            const countInput = await rl.question('📊 Número de resultados para busca (3): ') || '3';
            await semanticSearch(query, parseInt(countInput));
        }
    } finally {
        rl.close();
        console.log('\n🛑 Sessão encerrada');
    }
}

main().catch(err => {
    console.error('\n❌ Falha no sistema:', err);
    process.exit(1);
});