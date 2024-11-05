from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec, PodSpec  
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import bs4
import os
import time
import asyncio
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = 'end-to-end-rag'

class RAG:
    def __init__(self, web_url):
        loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Import nemoguardrails here after event loop setup
        from nemoguardrails import LLMRails, RailsConfig
        from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

        self.vectorstore_index_name = "end-to-end-rag"
        self.loader = WebBaseLoader(
            web_paths=(web_url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        self.model_name = "BAAI/bge-small-en"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name, model_kwargs=self.model_kwargs, encode_kwargs=self.encode_kwargs
        )
        # self.embeddings = SentenceTransformer('sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja')
        
        self.groq_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"), 
            model="llama3-70b-8192", 
            temperature=0
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=100
        )
        self.create_pinecone_index(self.vectorstore_index_name)
        self.vectorstore = PineconeVectorStore(
            index_name=self.vectorstore_index_name,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        self.rag_prompt = hub.pull(
            "rlm/rag-prompt", 
            api_key=os.getenv("LANGSMITH_API_KEY")
        )
        config = RailsConfig.from_path("./config")

        self.guardrails = RunnableRails(config=config, llm=self.groq_llm)


    def create_pinecone_index(self, vectorstore_index_name):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  
        spec = ServerlessSpec(cloud='aws', region='us-east-1')  
        if vectorstore_index_name in pc.list_indexes().names():  
            pc.delete_index(vectorstore_index_name)  
        pc.create_index(  
            vectorstore_index_name,  
            dimension=384,  # Match embedding dimension
            metric='dotproduct',  
            spec=spec  
        )  
        while not pc.describe_index(vectorstore_index_name).status['ready']:  
            time.sleep(1) 

    def load_docs_into_vectorstore_chain(self):
        docs = self.loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        
        documents_with_embeddings = [
            {
                "content": doc,
                "embedding": self.embeddings.embed_documents([doc.page_content])[0]
            }
            for doc in split_docs
        ]
        
        self.vectorstore.add_documents(documents_with_embeddings)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_retrieval_chain(self):
        self.load_docs_into_vectorstore_chain()
        self.retriever = self.vectorstore.as_retriever()
        self.rag_chain = (
            {
                "context": self.retriever | self.format_docs, "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.groq_llm
            | StrOutputParser()
        )
        self.rag_chain = self.guardrails | self.rag_chain
    
    def qa(self, query, vectorstore_created):
        if vectorstore_created:
            pass
        else:
            self.create_retrieval_chain()
        return self.rag_chain.invoke(query), True
