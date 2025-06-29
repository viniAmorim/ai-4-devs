// redis-store.ts
import { RedisVectorStore } from '@langchain/redis'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import { createClient } from 'redis'
import dotenv from 'dotenv'

dotenv.config()

if (!process.env.HUGGINGFACEHUB_API_TOKEN?.startsWith('hf_')) {
  throw new Error('Token do Hugging Face inválido ou ausente no .env')
}

const redisClient = createClient({ // Renomeado para evitar conflito com a exportação
  url: process.env.REDIS_URL || 'redis://localhost:6379'
})

// --- Instância para documentos JSON (seu atual) ---
export const jsonVectorStore = new RedisVectorStore(
  new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_TOKEN,
    model: process.env.EMBEDDINGS_MODEL,
    maxRetries: 3,
    timeout: 10000
  }),
  {
    indexName: 'artigos-embeddings', // Nome do índice original
    redisClient: redisClient, // Usa o cliente Redis compartilhado
    keyPrefix: 'artigos:'
  }
)

// --- NOVA Instância para documentos PDF ---
export const pdfVectorStore = new RedisVectorStore(
  new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_TOKEN,
    model: process.env.EMBEDDINGS_MODEL,
    maxRetries: 3,
    timeout: 10000
  }),
  {
    indexName: 'artigos-pdf-embeddings', // Nome do índice para PDFs (deve ser o mesmo do loader-pdf.ts)
    redisClient: redisClient, // Usa o cliente Redis compartilhado
    keyPrefix: 'artigos-pdf:'
  }
)

export { redisClient } // Exporta o cliente Redis para ser usado nos loaders e search