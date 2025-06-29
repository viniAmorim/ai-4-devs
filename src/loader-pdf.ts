import path from 'node:path'
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory'
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf' // Importa o PDFLoader
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter' // Um splitter mais adequado para PDFs
import { RedisVectorStore } from '@langchain/redis'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import { createClient } from 'redis'
import dotenv from 'dotenv'

dotenv.config()

// Diretório onde seus arquivos PDF estão
const PDF_DIRECTORY = path.resolve(__dirname, '../tmp/pdfs') // Exemplo: crie uma pasta 'pdfs' dentro de 'tmp'

const loader = new DirectoryLoader(
  PDF_DIRECTORY,
  {
    '.pdf': (path: string) => new PDFLoader(path) // Usa o PDFLoader para arquivos .pdf
  }
)

async function loadPdfs() {
  let redis
  try {
    console.log('🔄 Carregando documentos PDF...')
    const docs = await loader.load()
    console.log(`✅ ${docs.length} documentos PDF carregados para processamento inicial.`)

    // O RecursiveCharacterTextSplitter é geralmente mais robusto para PDFs
    // pois tenta manter a estrutura textual.
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500, // Ajuste este valor para o tamanho ideal do chunk para PDFs
      chunkOverlap: 100 // Sobreposição para garantir contexto entre chunks
    })

    const splittedDocs = await splitter.splitDocuments(docs)
    console.log(`✅ Documentos divididos em ${splittedDocs.length} chunks.`)

    // Adicione isso para inspecionar os primeiros chunks
    console.log('\n--- Conteúdo dos primeiros 3 chunks para inspeção ---');
    splittedDocs.slice(0, 3).forEach((chunk, index) => {
      console.log(`\n--- Chunk ${index + 1} ---`);
      console.log(chunk.pageContent);
      console.log(`Metadados: ${JSON.stringify(chunk.metadata)}`);
    });
    console.log('--- Fim da inspeção ---');

    redis = createClient({ 
      url: process.env.REDIS_URL || 'redis://localhost:6379'
    })
    
    await redis.connect()

    console.log('🔵 Criando embeddings para documentos PDF...')
    
    const embeddings = new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HUGGINGFACEHUB_API_TOKEN,
      model: process.env.EMBEDDINGS_MODEL,
      maxRetries: 3, 
      timeout: 10000 
    })

    await RedisVectorStore.fromDocuments(
      splittedDocs,
      embeddings,
      {
        indexName: 'artigos-pdf-embeddings', // Nome do índice diferente para PDFs
        redisClient: redis,
        keyPrefix: 'artigos-pdf:' // Prefixo de chave diferente
      }
    )

    console.log('✅ Documentos PDF carregados no Redis com sucesso!')
  } catch (error) {
    console.error('❌ Erro ao carregar documentos PDF:', error)
  } finally {
    if (redis) {
      console.log('Desconectando do Redis...')
      await redis.disconnect()
      console.log('Desconectado.')
    }
  }
}

loadPdfs()