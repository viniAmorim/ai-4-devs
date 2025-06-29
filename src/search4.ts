// search3.ts
import { RedisVectorStore } from '@langchain/redis';
// Importa ambas as instâncias e o cliente Redis
import { jsonVectorStore, pdfVectorStore, redisClient } from './redis-store';
import { PromptTemplate } from '@langchain/core/prompts';
import { HuggingFaceInference } from '@langchain/community/llms/hf';
import dotenv from 'dotenv';
import readline from 'node:readline/promises';

dotenv.config();

// ... (LLM_PROMPT e resultsPrompt permanecem os mesmos) ...
const LLM_PROMPT = PromptTemplate.fromTemplate(`[INST]
Você é um assistente de IA focado em fornecer respostas precisas, concisas e úteis com base EXCLUSIVAMENTE no contexto fornecido.
Não utilize conhecimento externo.
Se a informação necessária para responder à pergunta não estiver explicitamente no contexto, responda clara e diretamente que "Não foi possível encontrar informações relevantes para a sua consulta na nossa base de dados."

Contexto:
{context}

Pergunta: {query}

Com base no contexto fornecido, responda à pergunta de forma concisa e coerente.
[/INST]`);

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


async function semanticSearch(query: string, k: number = 3, searchType: 'json' | 'pdf' = 'json') { // Adiciona searchType
    let llmResponse = "";
    try {
        console.log(`\n🔍 Iniciando busca por: "${query}" no índice de ${searchType.toUpperCase()}...`);

        if (!redisClient.isOpen) { // Usa o redisClient exportado
            await redisClient.connect();
        }

        let currentVectorStore: RedisVectorStore;
        if (searchType === 'pdf') {
            currentVectorStore = pdfVectorStore;
        } else {
            currentVectorStore = jsonVectorStore;
        }
        
        let results: [any, number][] = [];
        let embeddingModelUsed = process.env.EMBEDDINGS_MODEL || 'API Padrão';

        try {
            results = await currentVectorStore.similaritySearchWithScore(query, k);
        } catch (apiError) {
            console.warn('⚠️ Alternando para embeddings locais...');
            const { HuggingFaceTransformersEmbeddings } = await import('@langchain/community/embeddings/hf_transformers');
            embeddingModelUsed = 'Xenova/all-MiniLM-L6-v2 (local)';

            const localEmbeddings = new HuggingFaceTransformersEmbeddings({
                modelName: embeddingModelUsed
            });

            // Se for necessário usar o RedisVectorStore local, precisa recriá-lo com o cliente redis
            const localStore = new RedisVectorStore(localEmbeddings, {
                redisClient: redisClient, // Usa o redisClient exportado
                indexName: currentVectorStore.indexName // Mantém o mesmo indexName da store original
            });

            results = await localStore.similaritySearchWithScore(query, k);
        }

        const RELEVANCE_THRESHOLD = 0.4;
        const filteredResults = results.filter(([, score]) => score >= RELEVANCE_THRESHOLD);

        let formattedRawResultsForDebug = '';
        let contextForLLM = '';

        if (filteredResults.length === 0) {
            formattedRawResultsForDebug = 'Nenhum documento relevante encontrado após filtragem.';
            llmResponse = "Não foi possível encontrar informações relevantes para a sua consulta na nossa base de dados.";
        } else {
            formattedRawResultsForDebug = filteredResults.map(([doc, score], index) => (
                `\n### Documento ${index + 1} (Score: ${score.toFixed(3)})\n` +
                `**Título:** ${doc.metadata?.titulo || 'Sem título'}\n` +
                `**Trecho:** ${doc.pageContent.substring(0, 300)}${doc.pageContent.length > 300 ? '...' : ''}`
            )).join('\n');

            contextForLLM = filteredResults.map(([doc, score], index) => (
                `Documento ${index + 1}:\n${doc.pageContent}`
            )).join('\n\n');

            const chatModelName = process.env.CHAT_MODEL;
            const huggingFaceApiKey = process.env.HUGGINGFACEHUB_API_TOKEN; 

            if (!chatModelName || !huggingFaceApiKey) {
                console.error("Variáveis de ambiente CHAT_MODEL ou HUGGINGFACEHUB_API_TOKEN não configuradas. Não será possível gerar respostas.");
                llmResponse = "Erro: Modelo de chat ou chave API não configurados para gerar respostas.";
            } else {
                const huggingFaceLLM = new HuggingFaceInference({
                    model: chatModelName,
                    apiKey: huggingFaceApiKey,
                    maxTokens: 300, 
                    temperature: 0.3, 
                    maxRetries: 5, // Tentar até 5 vezes em caso de falha temporária
                    timeout: 60000 // Aumenta o timeout para 60 segundos (o padrão é 10 segundos), dando mais tempo para o modelo "acordar"
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

                    const cleanGeneratedText = generatedText.split('[/INST]').pop()?.trim() || generatedText.trim();
                    llmResponse = cleanGeneratedText;

                } catch (llmError: any) {
                    console.error("❌ Erro ao chamar o LLM:", llmError.message || llmError);
                    llmResponse = "Houve um erro técnico ao gerar a resposta. Por favor, tente novamente mais tarde.";
                }
            }
        }

        console.log('\n✅ Resposta Gerada:');
        console.log(llmResponse);

        console.log('\n--- Painel de Análise (Detalhes Internos) ---');
        const analysisPrompt = await resultsPrompt.format({
            query,
            rawCount: results.length,
            embeddingModel: embeddingModelUsed,
            results: formattedRawResultsForDebug
        });
        console.log(analysisPrompt);

        console.log('\n🔍 Detalhes Técnicos Finais:');
        console.log(`Modelo LLM: ${process.env.CHAT_MODEL || 'Não configurado'}`);
        console.log(`Modelo Embedding: ${embeddingModelUsed}`);
        console.log(`Tempo: ${new Date().toLocaleTimeString()}`);

    } finally {
        // A desconexão será gerenciada no main() agora
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

        // Conecta o cliente Redis uma vez no início
        if (!redisClient.isOpen) {
            await redisClient.connect();
            console.log('🔌 Conectado ao Redis.');
        }

        while (true) {
            const query = await rl.question('\n🔎 Digite sua consulta (ou "sair"): ');
            if (query.toLowerCase() === 'sair') break;

            const countInput = await rl.question('📊 Número de resultados para busca (3): ') || '3';
            let searchTypeInput = await rl.question('📚 Buscar em (json/pdf - padrão json): ') || 'json';
            searchTypeInput = searchTypeInput.toLowerCase();

            if (searchTypeInput !== 'json' && searchTypeInput !== 'pdf') {
                console.warn('Tipo de busca inválido. Usando "json" como padrão.');
                searchTypeInput = 'json';
            }

            await semanticSearch(query, parseInt(countInput), searchTypeInput as 'json' | 'pdf');
        }
    } finally {
        rl.close();
        if (redisClient.isOpen) { // Garante a desconexão no final da sessão
            console.log('\nDesconectando do Redis...');
            await redisClient.disconnect();
        }
        console.log('\n🛑 Sessão encerrada');
    }
}

main().catch(err => {
    console.error('\n❌ Falha no sistema:', err);
    process.exit(1);
});