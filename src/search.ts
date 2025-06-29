import { RedisVectorStore } from '@langchain/redis'
import { redisVectorStore } from './redis-store'
import dotenv from 'dotenv'
import readline from 'node:readline/promises'

dotenv.config()

async function secureSearch(query: string, k: number = 3) {
  try {
    console.log(`\n🔍 Buscando: "${query}"`)
    
    // Conexão com verificação
    if (!redisVectorStore.redisClient.isOpen) {
      await redisVectorStore.redisClient.connect()
    }

    // Busca com fallback
    let results
    try {
      results = await redisVectorStore.similaritySearchWithScore(query, k)
    } catch (apiError) {
      console.warn('⚠️ Fallback para modelo local...')
      const { HuggingFaceTransformersEmbeddings } = await import('@langchain/community/embeddings/hf_transformers')
      const localEmbeddings = new HuggingFaceTransformersEmbeddings({
        modelName: 'Xenova/all-MiniLM-L6-v2'
      })
      
      const localStore = new RedisVectorStore(localEmbeddings, {
        redisClient: redisVectorStore.redisClient,
        indexName: 'artigos-embeddings'
      })
      
      results = await localStore.similaritySearchWithScore(query, k)
    }

    // Exibe resultados
    results.forEach(([doc, score], i) => {
      console.log(`\n📌 Resultado ${i + 1} (Score: ${score.toFixed(3)})`)
      console.log(doc.pageContent)
    })

  } finally {
    if (redisVectorStore.redisClient.isOpen) {
      await redisVectorStore.redisClient.disconnect()
    }
  }
}

// Interface interativa segura
async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  })

  try {
    console.log('\n=== Buscador Semântico ===')
    console.log(`Modo: ${process.env.HUGGINGFACEHUB_API_TOKEN ? 'API' : 'Local'}`)
    
    while (true) {
      const query = await rl.question('\nPergunta (ou "sair"): ')
      if (query.toLowerCase() === 'sair') break
      
      await secureSearch(query)
    }
  } finally {
    rl.close()
  }
}

main().catch(err => {
  console.error('Erro crítico:', err)
  process.exit(1)
})