// import { RedisVectorStore } from 'langchain/vectorstores/redis'
// import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
// import { createClient } from 'redis'

// export const redis = createClient({
//     url: 'redis://127.0.0.1:6379'
// })

// export const redisVectoreStore = new RedisVectorStore( 
//     new OpenAIEmbeddings({ openAIApisKey: process.env.OPENAI_API_KEY },
//         {
//             indexName: 'artigos-embeddings',
//             redisClient: redis,
//             keyPrefix: 'artigos:'
//         }
//     ),
//   )

// import { RedisVectorStore } from '@langchain/redis'
// import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
// import { createClient } from 'redis'

// export const redis = createClient({
//   url: process.env.REDIS_URL
// })

// export const redisVectorStore = new RedisVectorStore(
//   new HuggingFaceInferenceEmbeddings({
//     apiKey: process.env.HUGGINGFACEHUB_API_TOKEN,
//     model: process.env.EMBEDDINGS_MODEL
//   }),
//   {
//     indexName: 'artigos-embeddings',
//     redisClient: redis,
//     keyPrefix: 'artigos:'
//   }
// )


import { RedisVectorStore } from '@langchain/redis'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import { createClient } from 'redis'
import dotenv from 'dotenv'

dotenv.config()

// Verificação inicial da chave
if (!process.env.HUGGINGFACEHUB_API_TOKEN?.startsWith('hf_')) {
  throw new Error('Token do Hugging Face inválido ou ausente no .env')
}

const redis = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
})

export const redisVectorStore = new RedisVectorStore(
  new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_TOKEN,
    model: process.env.EMBEDDINGS_MODEL,
    maxRetries: 3,  // Adiciona retry automático
    timeout: 10000  // 10 segundos
  }),
  {
    indexName: 'artigos-embeddings',
    redisClient: redis,
    keyPrefix: 'artigos:'
  }
)