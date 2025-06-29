import { HuggingFaceInference } from '@langchain/community/llms/hf'
import { PromptTemplate } from '@langchain/core/prompts'
import { RetrievalQAChain } from 'langchain/chains'
import { redisVectorStore } from './redis-store'
import dotenv from 'dotenv'

dotenv.config()

// Modelos suportados em ordem de prioridade
const SUPPORTED_MODELS = [
  'mistralai/Mixtral-8x7B-Instruct-v0.1',  // Modelo alternativo
  'gpt2',                                  // Modelo fallback garantido
  'local'                                  // Ollama local
]

async function createLLM() {
  const apiKey = process.env.HUGGINGFACEHUB_API_TOKEN
  
  // Tentativa com Hugging Face
  if (apiKey?.startsWith('hf_')) {
    for (const model of SUPPORTED_MODELS) {
      try {
        if (model === 'local') break
        
        const llm = new HuggingFaceInference({
          apiKey,
          model,
          temperature: 0.3,
          maxRetries: 1,
          timeout: 15000
        })
        
        // Testa a conexão
        await llm.call('Teste de conexão')
        console.log(`\n🟢 Modelo selecionado: ${model}`)
        return llm
      } catch (error) {
        console.warn(`⚠️ Modelo ${model} indisponível: ${error.message}`)
      }
    }
  }

  // Fallback para Ollama local
  try {
    const { ChatOllama } = await import('@langchain/community/chat_models/ollama')
    console.log('\n🔵 Usando modelo local (Ollama)')
    return new ChatOllama({
      baseUrl: 'http://localhost:11434',
      model: 'mistral',
      temperature: 0.3
    })
  } catch (error) {
    throw new Error('Nenhum modelo disponível (nem Hugging Face, nem Ollama)')
  }
}

// Template otimizado
const prompt = PromptTemplate.fromTemplate(`
  Baseado nestes artigos:
  {context}

  Responda de forma técnica:
  {question}

  Requisitos:
  - Máximo 3 parágrafos
  - Formato markdown
  - Se não souber: "Sem informações suficientes"`)

async function queryDocuments(question: string) {
  const llm = await createLLM()
  const chain = RetrievalQAChain.fromLLM(llm, redisVectorStore.asRetriever(3), {
    prompt,
    returnSourceDocuments: true
  })

  try {
    console.log('\n🔗 Conectando ao Redis...')
    await redisVectorStore.redisClient.connect()

    console.log(`\n🔍 Processando: "${question}"`)
    const response = await chain.invoke({ query: question })
    
    console.log('\n💡 Resposta:')
    console.log(response.text)
    
    if (response.sourceDocuments?.length > 0) {
      console.log('\n📖 Fontes consultadas:')
      response.sourceDocuments.forEach((doc, i) => {
        console.log(`\n📌 Fonte ${i + 1}:`)
        console.log(`Título: ${doc.metadata?.titulo || 'Sem título'}`)
        console.log(`Conteúdo: ${doc.pageContent.substring(0, 120)}...`)
      })
    }

  } finally {
    if (redisVectorStore.redisClient?.isOpen) {
      await redisVectorStore.redisClient.disconnect()
    }
  }
}

// Execução com tratamento de erros completo
queryDocuments('Quais as vantagens do LangChain?')
  .catch(err => {
    console.error('\n❌ Falha crítica:', err.message)
    console.log('\nSoluções possíveis:')
    console.log('1. Obtenha uma chave em https://huggingface.co/settings/tokens')
    console.log('2. Instale Ollama local: curl -fsSL https://ollama.com/install.sh | sh')
    console.log('3. Use um modelo mais simples alterando SUPPORTED_MODELS')
    process.exit(1)
  })