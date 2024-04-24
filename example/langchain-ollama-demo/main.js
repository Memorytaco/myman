import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { input as prompts } from '@inquirer/prompts';

// application settings
const OLLAMA_BASE_URL = {
  baseUrl: "http://127.0.0.1:11434"
}

// fetch userguide documents
const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/user_guide"
);
const docs = await loader.load();

// split docs into small fragments
const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs);

// make embeddings
const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",  // use `ollama pull nomic-embed-text` before run this demo
  maxConcurrency: 5,
  ...OLLAMA_BASE_URL
});

// build retriever from vector store
const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);
const retriever = vectorstore.asRetriever();

// prepare the model, we use local ollama
const llm = new ChatOllama({
  model: "llama3",
  ...OLLAMA_BASE_URL
});

// define prompt template for chatting
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user's questions based on the below context:\n\n{context}",
  ],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);


// document chain
const docsChain = await createStuffDocumentsChain({
  llm: llm,
  prompt
});

// chat history chain
const historyAwarePrompt = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  [
    "user",
    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
  ],
]);
const historyAwareRetrieverChain = await createHistoryAwareRetriever({
  llm, retriever,
  rephrasePrompt: historyAwarePrompt,
});

// retriever chain
const retrievalChain = await createRetrievalChain({
  combineDocsChain: docsChain,
  retriever: historyAwareRetrieverChain,
});

// build a chat loop
let chat_history = [];
console.log(
`use ":exit" to quit and enjoy your journey
`)
while (true) {
  let input = await prompts({message: ">> "})
  if (input == ":exit" || input == "" || input == undefined) {
    break;
  }
  chat_history.push(new HumanMessage(input))
  const stream = await retrievalChain.stream({
    chat_history, input
  })
  let answer = ""
  for await (const chunk of stream) {
    if (chunk?.answer !== undefined) {
      process.stdout.write(chunk.answer)
      answer += chunk.answer
    }
  }
  console.log("") // flush with newline
  chat_history.push(new AIMessage(answer))
}

