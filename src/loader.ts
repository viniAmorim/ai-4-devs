import path from 'node:path'
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory'
import { JSONLoader } from 'langchain/document_loaders/fs/json'
import { TokenTextSplitter } from 'langchain/text_splitter'
import { RedisVectorStore } from '@langchain/redis'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import { createClient } from 'redis'
import dotenv from 'dotenv'

dotenv.config()

const loader = new DirectoryLoader(
  path.resolve(__dirname, '../tmp'),
  {
    '.json': (path: string) => new JSONLoader(path, '/conteudo')
  }
)

async function load() {
  let redis
  try {
    const docs = await loader.load()
    const splitter = new TokenTextSplitter({
      chunkSize: 600,
      chunkOverlap: 50
    })

    const splittedDocs = await splitter.splitDocuments(docs)
    redis = createClient({ 
      url: process.env.REDIS_URL || 'redis://localhost:6379'
    })
    
    await redis.connect()

    console.log('🔵 Criando embeddings...')
    
    const embeddings = new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HUGGINGFACEHUB_API_TOKEN,
      model: process.env.EMBEDDINGS_MODEL
    })

    await RedisVectorStore.fromDocuments(
      splittedDocs,
      embeddings,
      {
        indexName: 'artigos-embeddings',
        redisClient: redis,
        keyPrefix: 'artigos:'
      }
    )

    console.log('✅ Documentos carregados no Redis!')
  } catch (error) {
    console.error('❌ Erro ao carregar documentos:', error)
  } finally {
    if (redis) await redis.disconnect()
  }
}

load()