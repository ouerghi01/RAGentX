from langchain_community.document_loaders import CassandraLoader
from fastapi.concurrency import run_in_threadpool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
<<<<<<< HEAD
=======
from typing import Annotated
>>>>>>> 920b59ce2e8fef0c8c0a4cd9b6029e60bcfc2f7c
from dotenv import load_dotenv
import os 
from pathlib import Path
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains import RetrievalQA
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
from flashrank import Ranker 
class AgentInterface:
    def __init__(self,role):
        
        self.role=role
        self.UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR"))  
        self.MODEL_NAME_llm=os.getenv("MODEL_NAME")
        self.BASE_URL_OLLAMA=os.getenv("OLLAMA_BASE_URL")
        self.MODEL_NAME_EMBEDDING=os.getenv("MODEL_NAME_EMBEDDING")
        self.MODEL_KWARGS_EMBEDDING={"device": "cpu"}
        self.ENCODE_KWARGS={"normalize_embeddings": True}
        Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=3)
        self.CassandraInterface=CassandraInterface()
        self.hf_embedding = HuggingFaceEmbeddings(
        model_name=self.MODEL_NAME_EMBEDDING,  
        model_kwargs=self.MODEL_KWARGS_EMBEDDING,  
        encode_kwargs=self.ENCODE_KWARGS,
        )
        self.semantic_chunker = SemanticChunker (
        self.hf_embedding, 
        )
        self.documents=self.load_pdf_documents(True)
        self.CassandraInterface.clear_tables()
        self.bm25_retriever=BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k=2 # Retrieve top 2 results
        self.cassandra_vecstore :Cassandra = Cassandra(embedding=self.hf_embedding, session=self.CassandraInterface.session, keyspace=self.CassandraInterface.KEYSPACE, table_name="vectores_new")
        self.cassandra_vecstore.add_documents(self.documents)
        self.retrieval = self.cassandra_vecstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})
        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.retrieval],
                                        weights=[0.4, 0.6])
        self.compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=self.ensemble_retriever
        )
<<<<<<< HEAD
=======
        #https://aistudio.google.com/app/u/2/apikey
>>>>>>> 920b59ce2e8fef0c8c0a4cd9b6029e60bcfc2f7c
        self.llm_gemniu = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=os.getenv("LANGSMITH_API_KEY"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
<<<<<<< HEAD
=======

>>>>>>> 920b59ce2e8fef0c8c0a4cd9b6029e60bcfc2f7c
        self.llm=Ollama(model=self.MODEL_NAME_llm, base_url=self.BASE_URL_OLLAMA,verbose=True,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),)
        self.retrieval_chain=self.build_q_a_process()
    # Pre-retrieval Query Rewriting Function
    def answer_rewriting(self,context, query: str) -> str:
        query_rewrite_prompt = """
            You are a helpful assistant that takes a user's question and 
            provided context to generate a clear and direct answer. 
            Please provide a concise response based on the context without adding any extra comments.
            provide a short and clear answer based on the context provided.

            Context: {context}

            Question: {query}

            Answer:
        """

        QA_CHAIN_PROMPT = PromptTemplate(template=query_rewrite_prompt, input_variables=["query"])
        llm_chain = LLMChain(
            llm=self.llm, 
            prompt=QA_CHAIN_PROMPT, 
            callbacks=None, 
            verbose=True
        )
        retrieval_query = llm_chain.invoke({"query": query, "context": context})
        return retrieval_query['text']
    def build_q_a_process(self):

    
   
        prompt = """
        Je suis un agent intelligent qui répond uniquement en se basant sur les informations contenues dans la documentation fournie. Mon objectif est de fournir des réponses précises, claires et concises, sans ajouter d'informations extérieures.

        📌 Directives :
        1. **Utilisation exclusive du contexte** : Répondre uniquement à partir du contexte fourni.
        2. **Gestion des inconnues** : Si la réponse n'est pas présente dans la documentation, répondre : "Je ne sais pas."
        3. **Clarté et concision** : Les réponses doivent être brèves (maximum trois phrases) et directes.
        4. **Explication des fonctionnalités** : Expliquer brièvement tout outil, fonctionnalité ou concept mentionné dans le contexte.
        5. **Pas de suppositions** : Ne pas inventer ou extrapoler des informations non fournies.
        6. **Suggestion utile** : Donner, si possible, une recommandation pertinente basée sur le contexte.

        📖 Contexte :  
        {context}  

        ❓ Question :  
        {question}  

        ✍️ Réponse :

        """

        QA_CHAIN_PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question"])

<<<<<<< HEAD
        try:
            try:
                llm_chain = LLMChain(
                    llm=self.llm_gemniu, 
                    prompt=QA_CHAIN_PROMPT, 
                    callbacks=None, 
                    verbose=True
                )
            except Exception as e:
                print(f"Error with Gemini API: {e}")
                llm_chain = LLMChain(
                    llm=self.llm,  # Fallback to Ollama
                    prompt=QA_CHAIN_PROMPT, 
                    callbacks=None, 
                    verbose=True
                )
        except Exception as e:
            print(e)
            llm_chain = LLMChain(
            llm=self.llm, 
=======
        llm_chain = LLMChain(
            llm=self.llm_gemniu, 
>>>>>>> 920b59ce2e8fef0c8c0a4cd9b6029e60bcfc2f7c
            prompt=QA_CHAIN_PROMPT, 
            callbacks=None, 
            verbose=True
            )


        document_prompt = PromptTemplate(
            template="Context:\ncontent:{page_content}\nsource:{source}",  
            input_variables=["page_content", "source"]  
        )
        
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,  
            document_variable_name="context",
            callbacks=None,
            document_prompt=document_prompt  
        )
        qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,  
            retriever=self.compression_retriever,
            verbose=True
        )
        return qa
    def build_retrieval_chain_history(self):
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
        self.llm, self.compression_retriever, contextualize_q_prompt
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
        if bool==True:
            self.UPLOAD_DIR.mkdir(exist_ok=True) 
            docs=[]
            dir_path = Path("C:/Users/Ons/IA-Ollama-RAG-IMPL/src/uploads")
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
            
        #self.load_documents_from_cassandra(documents)
                
        return documents
    
    def load_documents_from_cassandra(self, documents):
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
    def answer_question(self,question:str,request,session_id):
        exist_answer=self.CassandraInterface.find_same_reponse(question)
        if exist_answer is not None:
            return{"request": request, "answer": exist_answer.answer, "question": question,"partition_id":exist_answer.partition_id,"timestamp":exist_answer.timestamp,"evaluation":True}
        else:
            question_enhanced= question 
            #chat_history = self.CassandraInterface.get_chat_history(session_id)
            final_answer= self.retrieval_chain.run(question_enhanced)
            final_answer= self.answer_rewriting(final_answer,question)
            self.CassandraInterface.insert_answer(session_id,question,final_answer)
            return {"request": request, "answer": final_answer, "question": question,"evaluation":True}

