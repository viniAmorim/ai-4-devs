import { RedisVectorStore } from '@langchain/redis'
import { redisVectorStore } from './redis-store'
import { PromptTemplate } from '@langchain/core/prompts'
import dotenv from 'dotenv'
import readline from 'node:readline/promises'

dotenv.config()

// Configuração do prompt avançado
const resultsPrompt = PromptTemplate.fromTemplate(`
  ## Análise de Documentos

  ### Contexto da Busca:
  - Termo pesquisado: "{query}"
  - Número de resultados: {count}
  - Modelo utilizado: {model}

  ### Resultados Encontrados:
  {results}

  ### Instruções para Análise:
  1. Identifique os 3 principais insights
  2. Destaque termos-chave relevantes
  3. Avalie a consistência entre os documentos
  4. Caso a resposta não conste nos documentos, responda que não tem essa informação.
`)

async function semanticSearch(query: string, k: number = 3) {
  try {
    console.log(`\n🔍 Iniciando busca por: "${query}"`)

    if (!redisVectorStore.redisClient.isOpen) {
      await redisVectorStore.redisClient.connect()
    }

    let results: [any, number][] = [] // Definindo o tipo para 'results'
    let modelUsed = process.env.EMBEDDINGS_MODEL || 'API Padrão'

    try {
      results = await redisVectorStore.similaritySearchWithScore(query, k)
    } catch (apiError) {
      console.warn('⚠️ Alternando para embeddings locais...')
      const { HuggingFaceTransformersEmbeddings } = await import('@langchain/community/embeddings/hf_transformers')
      modelUsed = 'Xenova/all-MiniLM-L6-v2 (local)'

      const localEmbeddings = new HuggingFaceTransformersEmbeddings({
        modelName: modelUsed
      })

      const localStore = new RedisVectorStore(localEmbeddings, {
        redisClient: redisVectorStore.redisClient,
        indexName: 'artigos-embeddings'
      })

      results = await localStore.similaritySearchWithScore(query, k)
    }

    // --- Nova Lógica de Filtro de Relevância ---
    const RELEVANCE_THRESHOLD = 0.5; // Ajuste este valor conforme seus testes
    const filteredResults = results.filter(([, score]) => score < RELEVANCE_THRESHOLD);

    let formattedResults = '';
    let analysisPrompt = '';

    if (filteredResults.length === 0) {
      formattedResults = 'Nenhum documento relevante encontrado.';
      analysisPrompt = await resultsPrompt.format({
        query,
        count: 0, // Zero resultados
        model: modelUsed,
        results: "Não foi possível encontrar informações relevantes nos documentos para a sua consulta."
      });
    } else {
      // Formatação dos resultados
      formattedResults = filteredResults.map(([doc, score], index) => (
        `\n### Documento ${index + 1} (Score: ${score.toFixed(3)})\n` +
        `**Título:** ${doc.metadata?.titulo || 'Sem título'}\n` +
        `**Trecho:** ${doc.pageContent.substring(0, 150)}${doc.pageContent.length > 150 ? '...' : ''}`
      )).join('\n');

      // Gera o prompt analítico
      analysisPrompt = await resultsPrompt.format({
        query,
        count: filteredResults.length,
        model: modelUsed,
        results: formattedResults
      });
    }
    // --- Fim da Nova Lógica ---

    console.log('\n📊 Painel de Análise:')
    console.log(analysisPrompt)

    // Detalhes técnicos
    console.log('\n🔍 Detalhes Técnicos:')
    console.log(`Modelo: ${modelUsed}`)
    console.log(`Tempo: ${new Date().toLocaleTimeString()}`)

  } finally {
    if (redisVectorStore.redisClient.isOpen) {
      await redisVectorStore.redisClient.disconnect()
    }
  }
}

// Interface interativa melhorada
async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  })

  try {
    console.log('\n=== Sistema de Análise Semântica ===')
    console.log('📌 Modo:', process.env.HUGGINGFACEHUB_API_TOKEN ? 'API' : 'Local')

    while (true) {
      const query = await rl.question('\n🔎 Digite sua consulta (ou "sair"): ')
      if (query.toLowerCase() === 'sair') break

      const countInput = await rl.question('📊 Número de resultados (3): ') || '3'
      await semanticSearch(query, parseInt(countInput))
    }
  } finally {
    rl.close()
    console.log('\n🛑 Sessão encerrada')
  }
}

main().catch(err => {
  console.error('\n❌ Falha no sistema:', err)
  process.exit(1)
})