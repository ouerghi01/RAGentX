from langchain_community.document_loaders import CassandraLoader
from fastapi.concurrency import run_in_threadpool
from typing import Annotated
from dotenv import load_dotenv
import os 
from pathlib import Path
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
load_dotenv()  # take environment variables from .env.
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_experimental.text_splitter import SemanticChunker
from services.cassandra_service import CassandraInterface
class AgentInterface:
    def __init__(self,role):
        self.role=role
        self.UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR"))  
         
        self.MODEL_NAME_llm=os.getenv("MODEL_NAME")
        self.BASE_URL_OLLAMA=os.getenv("OLLAMA_BASE_URL")
        self.MODEL_NAME_EMBEDDING=os.getenv("MODEL_NAME_EMBEDDING")
        self.MODEL_KWARGS_EMBEDDING={"device": "cpu"}
        self.ENCODE_KWARGS={"normalize_embeddings": True}
        
        self.CassandraInterface=CassandraInterface()
        self.hf_embedding = HuggingFaceEmbeddings(
        model_name=self.MODEL_NAME_EMBEDDING,  
        model_kwargs=self.MODEL_KWARGS_EMBEDDING,  
        encode_kwargs=self.ENCODE_KWARGS,
        )
        self.semantic_chunker = SemanticChunker (
        self.hf_embedding, 
        )
        self.documents=self.load_pdf_documents(False)
        self.CassandraInterface.clear_tables()
        self.bm25_retriever=BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k=2 # Retrieve top 2 results
        self.cassandra_vecstore :Cassandra = Cassandra(embedding=self.hf_embedding, session=self.CassandraInterface.session, keyspace=self.CassandraInterface.KEYSPACE, table_name="vectores_new")
        self.cassandra_vecstore.add_documents(self.documents)
        self.retrieval = self.cassandra_vecstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})
        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.retrieval],
                                        weights=[0.4, 0.6])

        self.llm=Ollama(model=self.MODEL_NAME_llm, base_url=self.BASE_URL_OLLAMA)
        self.retrieval_chain=self.build_retrieval_chain()
    def build_retrieval_chain(self):
        contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which may reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. If no chat history exists, use "
        "the available context to understand and reformulate the question. "
        "Do NOT answer the question, just reformulate it if needed."
        "Consider both chat history and context when available, "
        "but be able to work with either one independently."
        )

        # Define the contextualization prompt template
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
        self.llm, self.ensemble_retriever, contextualize_q_prompt
        )
        qa_system_prompt = (
        f"You are part of a multi-agent system designed to answer questions. Your role is: {self.role} 🤖\n"
        "Each agent will contribute to answering the question based on specific parts of the context: \n"
        "When answering, structure the response clearly using bullet points or numbered lists. "
        "🧠 Analysis Agent - Breaks down the question\n"
        "📊 Expert Agent - Provides domain expertise\n" 
        "✅ Validation Agent - Verifies accuracy\n"
        "🔍 Research Agent - Explores context\n"
        "After gathering individual contributions, a final agent will combine and deliver a concise response in no more than three sentences. 📝\n"
        "At the end, provide a brief summary emphasizing the key factors that most affect the total loan cost. "
        "If an agent cannot provide an answer, it should respond with 'I don't know.' ❌"
            "Keep the explanation clear, avoiding unnecessary complexity while maintaining accuracy."
        )

        # Define the Q&A prompt template
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", "Context: {context}"),  
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return rag_chain
    def load_pdf_documents(self,bool=True):
        documents = []
        if bool==False:
            self.UPLOAD_DIR.mkdir(exist_ok=True) 
            docs=[]
            dir_path = Path("/home/aziz/IA-DeepSeek-RAG-IMPL/src/uploads")
            for file in os.listdir(dir_path):
                if file.endswith(".pdf"):
                    full_path = dir_path/ file
                    loader = PDFPlumberLoader(full_path)
                    document = loader.load()
                    docs.append(document)
            for doc in docs:
                if doc is not None:
                    chunks = self.semantic_chunker.split_documents(doc)
                    documents.extend(chunks)  
            
        schemas = self.CassandraInterface.retrieve_column_descriptions()
            
        for table_name,desc in schemas.items():
            loader=CassandraLoader(
                table=table_name,
                session=self.CassandraInterface.session,
                keyspace=self.CassandraInterface.KEYSPACE,
                )
            docs=loader.load()
            if table_name!="vectores":
                    for doc in docs:
                        if doc is not None:
                            doc.metadata['source'] = f'Description:{desc}.{table_name}'
            documents.extend(docs)
                
        return documents
    def answer_question(self,question:str,request,session_id):
        exist_answer=self.CassandraInterface.find_same_reponse(question)
        if exist_answer is not None:
            return{"request": request, "answer": exist_answer.answer, "question": question,"partition_id":exist_answer.partition_id,"timestamp":exist_answer.timestamp,"evaluation":True}
        else:
            chat_history = self.CassandraInterface.get_chat_history(session_id)
            final_answer= self.retrieval_chain.invoke({"input": question, "chat_history": chat_history})
            self.CassandraInterface.insert_answer(session_id,question,final_answer)
            return {"request": request, "answer": final_answer["answer"], "question": question,"evaluation":True}

