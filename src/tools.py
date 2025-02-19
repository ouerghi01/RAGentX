from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
import os
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
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
    cloud_config = {
    'secure_connect_bundle': 'C:/Users/Ons/IA-Ollama-RAG-IMPL/src/sec/secure-connect-store-base.zip'
    }

    with open("C:/Users/Ons/IA-Ollama-RAG-IMPL/src/sec/token.json") as f:
       secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    try:
        auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        session = cluster.connect()
    except Exception as e:
        session = Cluster(["localhost"],port=9042).connect( )
        cassio.init(session=session, keyspace="store_key")
        #print(f"Failed to connect to Cassandra: {str(e)}")
        
    
    create_table = """
    CREATE TABLE IF NOT EXISTS store_key.vectores (
        partition_id UUID PRIMARY KEY,
        document_text TEXT,  -- Text extracted from the PDF
        document_content BLOB,  -- PDF content stored as binary data
        vector BLOB  -- Store the embeddings (vector representation of the document)
    );
    """
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
session = initialize_database_session()
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
    keyspace = "store_key"
    cassandra_vecstore :Cassandra = Cassandra(embedding=hf, session=session, keyspace=keyspace, table_name="vectores_new")
    cassandra_vecstore.add_documents(documents)
    retrieval = cassandra_vecstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    return retrieval
def get_last_responses(session):
    query = "SELECT * FROM store_key.response_table LIMIT 10"
    rows = session.execute(query)
    text=""
    for row in rows:
        text+=f"Question: {row.question}\nAnswer: {row.answer}\nTimestamp: {row.timestamp}\nEvaluation: {row.evaluation}\n\n"
    return text
def build_q_a_process(retrieval, model_name="deepseek-r1:7b", llm_model="qwen2.5:3b", base_url="http://localhost:11434"):
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

    # Create a history-aware retriever for fetching relevant context
    history_aware_retriever = create_history_aware_retriever(
        llm, retrieval, contextualize_q_prompt
    )

    # Define the multi-agent system for answering the question
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
            ("system", "Context: {context}"),  # Ensure proper context is passed
            ("human", "{input}"),
        ]
    )

    # Create the document chain for answering the question based on the context
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the RAG (Retrieval-Augmented Generation) chain
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
def fetch_relevant_documents_from_cassandra():
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
#session=initialize_database_session()

    session = Cluster(["0.0.0.0"],port=9042).connect( )

    if session is None:
        session=initialize_database_session()
#session.execute("TRUNCATE store_key.response_table")
    schema = session.execute("SELECT table_name FROM system_schema.tables WHERE keyspace_name='store_key';")
    schema_str = ""
    for i,row in enumerate(schema):
        if i==0:
            schema_str+=f"description table: table for storing all responses and questions    table_name: {row.table_name}\n"
        else:
            schema_str+=f"description table: table for all documents that guide users through the marketplace, table_name: {row.table_name}\n"
   
    question = "Comment puis-je valider ou refuser une offre dans l'interface contenant les détails de l'offre tels que la Désignation, la Date de création, la Date de début d'offre, la Date de fin d'offre, la Promotion et la Quantité d'offre ?"
    response_expected = """Pour valider l'offre, il suffit de cliquer sur le bouton d'action Valider.
Pour refuser l'offre, il suffit de cliquer sur le bouton d'action Refus."""
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
    return session,llm,question,response_expected,documents
    # Connect to Cassandra
    session = Cluster(["localhost"],port=9042).connect( )
    import cassio
    cassio.init(session=session, keyspace="store_key")

    # Create tables
    create_table = """
        CREATE TABLE IF NOT EXISTS store_key.vectores (
            partition_id UUID PRIMARY KEY,
            document_text TEXT,  -- Text extracted from the PDF
            document_content BLOB,  -- PDF content stored as binary data
            vector BLOB  -- Store the embeddings (vector representation of the document)
        );
        """
    response_table = """
        CREATE TABLE IF NOT EXISTS store_key.response_table (
            partition_id UUID PRIMARY KEY,
            question TEXT,  
            answer TEXT,
            timestamp TIMESTAMP,
            evaluation BOOLEAN
            
        );
        """

    # Test question and expected response
    question = "Comment puis-je valider ou refuser une offre dans l'interface contenant les détails de l'offre tels que la Désignation, la Date de création, la Date de début d'offre, la Date de fin d'offre, la Promotion et la Quantité d'offre ?"
    response_expected = """Pour valider l'offre, il suffit de cliquer sur le bouton d'action Valider.
    Pour refuser l'offre, il suffit de cliquer sur le bouton d'action Refus."""

    # Initialize LLM model   
    llm=Ollama(model="qwen2.5:3b", base_url="http://localhost:11434")
        
    # Execute table creation
    session.execute(create_table)
    session.execute(response_table)

    # Load and process documents
    docs= load_pdf_documents()
    Retrieval= create_retriever_from_documents(session, docs)
    retrieved_docs= Retrieval.get_relevant_documents(question)

    # Setup document prompt template
    document_prompt = PromptTemplate(
            template="Context:\ncontent:{page_content}\nsource:{source}",  
            input_variables=["page_content", "source"]  
        )

    # Format context with proper spacing
    context = "\n\n".join([document_prompt.format(page_content=" ".join(doc.page_content.split()), source=doc.metadata.get('source', 'N/A')) for doc in retrieved_docs])

    # Define main prompt template
    template = """
    Tu es un assistant IA intégré à un système multi-agent collaboratif, conçu pour fournir des réponses claires, précises et utiles. Chaque agent joue un rôle spécifique et coopère pour offrir la meilleure assistance possible. Respecte strictement les consignes suivantes :
        🔹 📜 Principes fondamentaux :
            - Ne t'appuie que sur les informations du contexte ci-dessous pour répondre.
            - Si une information est absente, indique-le explicitement avec "Je ne sais pas." et propose une solution pour l'obtenir.
            - Travaille en synergie avec les autres agents en te concentrant sur ta spécialisation et en complétant leurs contributions.
            - Fournis des réponses courtes et impactantes (maximum trois phrases), sans sacrifier la clarté ni la pertinence.
            - Si un outil ou une plateforme est mentionné(e), explique son objectif et son utilisation de manière simple et pratique.
            - Ne fais aucune supposition : toute réponse doit être directement appuyée par le contexte fourni.
            - Si pertinent, ajoute une recommandation actionnable pour aider l'utilisateur à mieux comprendre ou agir efficacement.
        🔹 🤖 Rôles des agents spécialisés :
            - 🧠 Agent_Analyse : Décompose la question et identifie les informations essentielles du contexte.
            - 📊 Agent_Expertise : Fournit une interprétation experte des données disponibles.
            - ✅ Agent_Validation : Vérifie la précision, la clarté et la pertinence de la réponse finale.
            - 🔍 Agent_Recherche : Explore le contexte pour repérer des informations cachées ou indirectement liées.
            - 🎯 Agent_Optimisation : Reformule la réponse pour la rendre plus concise, fluide et impactante.
            - 📌 Agent_Contextuel : Assure que la réponse respecte bien les contraintes spécifiques du contexte.
            - 💡 Agent_Recommandation : Ajoute des conseils ou des suggestions pratiques pour aider l'utilisateur à agir efficacement.
            - 🛠 Agent_Technique : Si la question concerne un outil, une plateforme ou une technologie, explique son fonctionnement de manière détaillée et pratique.
            - 📖 Agent_Pédagogique : Simplifie les concepts complexes et les explique de manière accessible.
        📂 Contexte :
        {context}
        ❓ Question :
        {question}
        📝 Réponse (collaborative) :
    """

    # Setup QA chain prompt
    QA_CHAIN_PROMPT = PromptTemplate(
            template=template,
            input_variables=[ "context","question"]
        )

    # Create LLM chain
    llm_chain = LLMChain(
            llm=llm, 
            prompt=QA_CHAIN_PROMPT, 
            callbacks=None, 
            verbose=True
        )

    # Get response
    response=llm_chain.run(context=context,question=question)

    # Text similarity comparison functions
    WORD = re.compile(r"\w+")

    def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)

    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    # Calculate similarity score
    response_expected_vector = text_to_vector(response_expected)
    response_vector = text_to_vector(response)
    cosine = get_cosine(response_expected_vector, response_vector)

    # Print results
    print("******************************************response******************************************")
    print(response)
    print("******************************************response_expected******************************************")
    print(response_expected)
    print("******************************************cosine******************************************")
    print(cosine)
    print("******************************************//////******************************************")

#session.execute("TRUNCATE store_key.vectores")
#docs=load_pdf_documents()
#retr= create_retriever_from_documents(session, docs)
#loader = CassandraLoader(
#    table="vectores",
#    session=session,
#    keyspace=CASSANDRA_KEYSPACE,
#)
#docs = loader.load()
#print(docs)




#from langchain.chains import ConversationalRetrievalChain
#from langchain.prompts.prompt import PromptTemplate

# Template setup
#template = """
#You are HR assistant to select best candidates based on the resume based on the user input. It is important to return resume ID when you find the promising resume. Start with AAAAAAAAAAAAA
#Here is context including list of resume information: {context}
#user input: {question} 
#AI Assistant: start with AAAAAAAAAAAAA
#[/INST]"""
#PROMPT = PromptTemplate(input_variables=["question", "context"], template=template)

# Chain initialization
#conversation_chain = ConversationalRetrievalChain.from_llm(
    #llm=llm,
    #retriever=db.as_retriever(search_kwargs={'k': 4}),
    #condense_question_prompt = PROMPT,
#)
