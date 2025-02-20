from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
import os
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from typing import Literal
from langchain.pydantic_v1 import BaseModel,Field
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime
import uvicorn
from langchain_community.document_loaders import CassandraLoader
from fastapi.concurrency import run_in_threadpool
from typing import Annotated
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
from fastapi import FastAPI 
from fastapi import FastAPI,Request,UploadFile,File,Form
import threading
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import uuid
import math
import re
import cassio

from collections import Counter
from datetime import datetime
from pathlib import Path
app = FastAPI()
model_name = "deepseek-r1:7b"
UPLOAD_DIR = Path("uploads")  
UPLOAD_DIR.mkdir(exist_ok=True)  
session = None
default_path_file = Path(r"uploads/Guide d'Utilisation store véhicule (1).pdf")
Retrieval = None
qa = None
templates = None

def initialize_database_session():
    
    session = Cluster(["localhost"],port=9042).connect( )
    cassio.init(session=session, keyspace="store_key")
        #print(f"Failed to connect to Cassandra: {str(e)}")
        
    create_key_space="""
    CREATE KEYSPACE IF NOT EXISTS store_key
    WITH replication = {'class':'SimpleStrategy', 'replication_factor' : 3};
    """
    session.execute(create_key_space)
    create_table = """
    CREATE TABLE IF NOT EXISTS store_key.vectores (
        partition_id UUID PRIMARY KEY,
        document_text TEXT,  -- Text extracted from the PDF
        document_content BLOB,  -- PDF content stored as binary data
        vector BLOB  -- Store the embeddings (vector representation of the document)
    );
    """
    create_session_table = """

    CREATE TABLE IF NOT EXISTS store_key.session_table (
        session_id UUID PRIMARY KEY,
        
    );
    """
    session.execute(create_session_table)
    session_response_table = """
    CREATE TABLE IF NOT EXISTS store_key.response_session (
        partition_id UUID PRIMARY KEY,
        session_id UUID,
        table_response_id UUID,
    );
    """
    session.execute(session_response_table)
    
    response_table = """
    CREATE TABLE IF NOT EXISTS store_key.response_table (
        partition_id UUID PRIMARY KEY,
        question TEXT,  
        answer TEXT,
        timestamp TIMESTAMP,
        evaluation BOOLEAN
        
    );
    """
    session.execute(create_table)
    session.execute(response_table)
    return session
#session = initialize_database_session()
def retrieve_column_descriptions(session):
    query_schema ="""
    SELECT * FROM system_schema.columns 
    WHERE keyspace_name = 'store_key' ;
    """
    rows=session.execute(query_schema)
    schema={}
    for row in rows:
        if schema.get(row.table_name) is None:
            schema[row.table_name]=f"description : \n column_name : {row.column_name}  type : {row.type} \n"
        else:
            schema[row.table_name]+=f" column_name : {row.column_name}  type : {row.type} \n"
    return schema



def load_pdf_documents():
    
    docs=[]
    p = Path("uploads")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for file in os.listdir(p):
        if file.endswith(".pdf"):
            full_path = dir_path/UPLOAD_DIR / file
            loader = PDFPlumberLoader(full_path)
            document = loader.load()
            docs.append(document)
    print(len(docs))
    return docs
    
def create_retriever_from_documents(session, docs):
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  
    # intfloat/e5-small is a smaller model that can be used for faster inference
    #model_name = "intfloat/e5-small"
    model_kwargs = {'device': 'cpu'}  # Use CPU for inference
    encode_kwargs = {'normalize_embeddings': True}  # Normalizing the embeddings for better comparison
    #HuggingFaceEmbeddings
    hf = HuggingFaceEmbeddings(
        model_name=model_name,  
        model_kwargs=model_kwargs,  
        encode_kwargs=encode_kwargs
    )
    text_splitter = SemanticChunker (
    hf 
    )
    documents = []
    for doc in docs:
        if doc is not None:
            chunks = text_splitter.split_documents(doc)
            documents.extend(chunks)  
    bm25_retriever=BM25Retriever.from_documents(documents)
    bm25_retriever.k =  2  # Retrieve top 2 results

    keyspace = "store_key"
    cassandra_vecstore :Cassandra = Cassandra(embedding=hf, session=session, keyspace=keyspace, table_name="vectores_new")
    #cassandra_vecstore.add_documents(documents)
    retrieval = cassandra_vecstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retrieval],
                                       weights=[0.4, 0.6])

    return ensemble_retriever
def get_last_responses(session):
    query = "SELECT * FROM store_key.response_table LIMIT 10"
    rows = session.execute(query)
    text=""
    for row in rows:
        text+=f"Question: {row.question}\nAnswer: {row.answer}\nTimestamp: {row.timestamp}\nEvaluation: {row.evaluation}\n\n"
    return text
def build_q_a_process(retrieval, model_name="deepseek-r1:7b", llm_model="qwen2.5:0.5b", base_url="http://localhost:11434"):
    """
    Build a Question-Answering process using a multi-agent system.
    
    Args:
        retrieval: The retrieval object used for fetching relevant context.
        model_name (str): Name of the model to be used for Q&A.
        llm_model (str): LLM model used for contextualization and answering.
        base_url (str): URL of the local LLM server.
        
    Returns:
        rag_chain: The Retrieval-Augmented Generation (RAG) chain for answering questions.
    """
    
    # Initialize the LLM for generating responses
    llm = Ollama(model=llm_model, base_url=base_url)

    # Contextualize the question based on chat history and the latest user input
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
        llm, retrieval, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are part of a multi-agent system designed to answer questions. 🤖\n"
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

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    return rag_chain


    
#question="Comment ajouter un modèle dans l'interface de gestion des modèles ?"
def execute_question_answering(model_name, initialize_database_session, load_pdf_documents, create_retriever_from_documents, build_q_a_process, question):
    docs=load_pdf_documents()
    session=initialize_database_session()
    Retrieval=create_retriever_from_documents(session,docs)
    retrieved_docs= Retrieval.get_relevant_documents(question)
    _,llm_chain,document_prompt=build_q_a_process(Retrieval,model_name)

# Ensure context formatting is correct by adding spaces between words
    context = "\n\n".join([document_prompt.format(page_content=" ".join(doc.page_content.split()), source=doc.metadata.get('source', 'N/A')) for doc in retrieved_docs])

    steps:str = llm_chain.run(context=context, question=question)
    try:
        steps=steps[steps.index("</think>")+8:]
        cleaned_steps = steps.replace("```json", "").replace("```", "").strip()
        parsed_steps = json.loads(cleaned_steps)
        reponses=[]
        llm=Ollama(model="gemma", base_url="http://localhost:11434")
        threads=[]
        for agent in parsed_steps["agents_tasks"]:
            role_agent=agent["agent"]
            task_agent=agent.get("task", agent.get("tache"))  # Ensure key consistency
            template="""
        I am an AI assistant specialized in providing clear, precise, and useful answers. I work in a multi-agent collaborative system designed to offer the best possible assistance. Here is my contribution to the collaborative response:
        Context: {context}
        Role: {role}
        Task: {task}
        """
            QA_CHAIN_PROMPT = PromptTemplate(
        template=template,
        input_variables=[ "role", "task","context"]
        )
        
            llm_chain = LLMChain(
            llm=llm, 
            prompt=QA_CHAIN_PROMPT, 
            callbacks=None, 
            verbose=True
        )
            def run_ollama():
                response=llm_chain.run(role=role_agent,task=task_agent,context=context)
                reponses.append(response)
            thread=threading.Thread(target=run_ollama)
            threads.append(thread)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        final_llm=Ollama(model="deepseek-r1:7b", base_url="http://localhost:11434")
        final_context = "\n\n".join(reponses)
        final_template="""
    I am an AI assistant specialized in providing clear, precise, and useful answers. I work in a multi-agent collaborative system designed to offer the best possible assistance. Here is my contribution to the collaborative response:
    role: your role is to provide a clear, precise, and useful answer from all the information provided in the context.
    Context: {context}
    question: {question}
           """
        QA_CHAIN_PROMPT = PromptTemplate(
    template=final_template,
    input_variables=[ "context", "question"]
    )
        llm_chain = LLMChain(
        llm=final_llm, 
        prompt=QA_CHAIN_PROMPT, 
        callbacks=None, 
        verbose=True
    )
        final_response=llm_chain.run(context=final_context,question=question)
        print(final_response)
    except json.JSONDecodeError as e:
        print("Raw output:", cleaned_steps)

#execute_question_answering(model_name, initialize_database_session, load_pdf_documents, create_retriever_from_documents, build_q_a_process, question)
def fetch_relevant_documents_from_cassandra(question):
    template = """
    I am an AI assistant specialized in providing clear, precise, and useful answers. 
    Your role is to decide which tables from the Cassandra database schema should be used based on the question provided.
    Instructions:
    - Review the provided schema
    - List only relevant table names
    - Separate multiple tables with commas
    - Do not include explanations
    - Do not provide additional information
    - Do not use indice
    response format: give the table names separated by commas
    Schema: 
    {schema}
    Question: 
    {question}
    Response:
"""
    QA_CHAIN_PROMPT = PromptTemplate(
        template=template,
        input_variables=[ "schema","question"]
        )
    llm=Ollama(model="qwen2.5:3b", base_url="http://localhost:11434")
    llm_chain = LLMChain(
        llm=llm, 
        prompt=QA_CHAIN_PROMPT, 
        callbacks=None, 
        verbose=True
    )

    
    schema = retrieve_column_descriptions(session)
    schema_str = "\n".join([f"Table: {table}\n{desc}" for table, desc in schema.items()])
    
    tabels_nedeed = llm_chain.run(schema=schema_str, question=question).split(",")
    documents=[]
    CASSANDRA_KEYSPACE = "store_key"

    for table in tabels_nedeed:
        table_name=table.strip()
        loader=CassandraLoader(
        table=table_name,
        session=session,
        keyspace=CASSANDRA_KEYSPACE,
        )
        docs=loader.load()
        if table_name=="response_table":
            for doc in docs:
                if doc is not None:
                    doc.metadata['source'] = f'cassandra:{CASSANDRA_KEYSPACE}.{table_name}'
        documents.extend(docs)
    return documents
   