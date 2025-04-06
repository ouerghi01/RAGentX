from langchain.retrievers import ContextualCompressionRetriever
import logging
import time
from collections import OrderedDict
from langchain_community.document_compressors import FlashrankRerank

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_core.globals import set_llm_cache
from fastapi.responses import HTMLResponse
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import TokenTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_astradb import AstraDBVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
import os 
from pathlib import Path
from langchain.retrievers import  EnsembleRetriever

from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel


from services.load_data import DataLoader

# We can do the same thing with a SQLite cache
from langchain_community.cache import SQLiteCache

from services.response_html import generate_answer_html

#from cql_agent import CQLAGENT
set_llm_cache(SQLiteCache(database_path="/home/aziz/IA-DeepSeek-RAG-IMPL/APP/langchain.db"))
load_dotenv()  # take environment variables from .env.
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_experimental.text_splitter import SemanticChunker
from services.cassandra_service import CassandraManager
from flashrank import Ranker 
from langchain_upstage import UpstageGroundednessCheck
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
class Answer(BaseModel):
    reponse: str = Field(description="The answer to the question")
class Answers(BaseModel):
    answers: list[Answer] = Field(description="List of answers to the questions")
parser = JsonOutputParser(pydantic_object=Answers)
class User(BaseModel):
    username: str
    email: str | None = None
    his_job: str | None = None
    hashed_password: str
class AgentInterface:
    """
    A comprehensive agent interface that manages document processing, retrieval, and question-answering capabilities.
    This class implements an advanced RAG (Retrieval-Augmented Generation) system that combines multiple
    retrieval strategies, embedding models, and language models to provide accurate responses to queries
    based on document content.
    Key Features:
    - Multiple document retrieval strategies (BM25, semantic search, parent-child)
    - Ensemble approach combining different retrievers
    - Support for PDF processing with OCR capabilities
    - Integration with various storage systems (Cassandra, AstraDB)
    - Conversation memory management
    - Multiple LLM support (Ollama, Gemini)
        role (str): The role specification for the agent
        UPLOAD_DIR (Path): Directory path for document uploads
        MODEL_NAME_llm (str): Name of the LLM model
        pdfservice (PDFService): Service for PDF operations
        astra_db_store (Union[AstraDBVectorStore, Cassandra]): Vector store for embeddings
        llm_gemini (ChatGoogleGenerativeAI): Gemini model instance
        documents (list): List of loaded documents
        parent_store (InMemoryStore): Storage for parent documents
        ensemble_retriever (EnsembleRetriever): Final ensemble retriever
        memory (ConversationSummaryMemory): Conversation history memory
        retrieval_chain: Configured retrieval chain for QA
        >>> agent = AgentInterface(role="technical_assistant")
        >>> answer = agent.answer_question("What is RAG?", request_obj, "session_123")
        Requires appropriate environment variables to be set for model names,
        API endpoints, and authentication tokens.
    """
    def __init__(self,role="assistant",cassandra_intra:CassandraManager=CassandraManager(),name_dir="/home/aziz/IA-DeepSeek-RAG-IMPL/APP/uploads"
        ):
        """
        Initialize the AgentService class with various components for document processing and retrieval.
        This service handles document processing, embedding, vector storage, and retrieval operations
        using multiple models and retrievers in an ensemble approach.
        Args:
            role (str): The role specification for the agent service
        Attributes:
            role (str): Stored role specification
            UPLOAD_DIR (Path): Directory path for uploading documents
            MODEL_NAME_llm (str): Name of the LLM model from environment variables
            BASE_URL_OLLAMA (str): Base URL for Ollama service
            MODEL_NAME_EMBEDDING (str): Name of the embedding model
            MODEL_KWARGS_EMBEDDING (dict): Configuration for embedding model
            ENCODE_KWARGS (dict): Configuration for encoding
            pdfservice (PDFService): Service for handling PDF operations
            CassandraInterface (CassandraInterface): Interface for Cassandra operations
            hf_embedding (HuggingFaceEmbeddings): Hugging Face embeddings model
            semantic_chunker (SemanticChunker): Chunker for semantic text splitting
            astra_db_store (AstraDBVectorStore|Cassandra): Vector store for document embeddings
            llm_gemini (ChatGoogleGenerativeAI): Google's Gemini model instance
            documents (list): Loaded PDF documents
            parent_store (InMemoryStore): In-memory storage for parent documents
            retrieval (BaseRetriever): Base retriever instance
            ensemble_retriever_new (EnsembleRetriever): Combined retriever instance
            multi_retriever (MultiQueryRetriever): Multi-query retrieval system
            ensemble_retriever (EnsembleRetriever): Final ensemble retrieval system
            compression_retriever (ContextualCompressionRetriever): Compression-enabled retriever
            combine_documents_chain (Optional): Chain for combining documents
            llm (Ollama): Ollama model instance
            memory (ConversationSummaryMemory): Memory system for conversation history
        Raises:
            Exception: If AstraDBVectorStore initialization fails, falls back to Cassandra
        """

        self.role=role
        self.setup_logging()
        self.prompt=None
        self.cache = OrderedDict()
        self.cache_ttl = 300
        self.name_dir=name_dir
        self.UPLOAD_DIR = Path(self.name_dir) 
        self.load_data=DataLoader(upload_dir=self.UPLOAD_DIR) 
        
        self.MODEL_NAME_llm=os.getenv("MODEL_NAME")
        self.BASE_URL_OLLAMA=os.getenv("OLLAMA_BASE_URL")
        self.MODEL_NAME_EMBEDDING=os.getenv("MODEL_NAME_EMBEDDING")
        self.MODEL_KWARGS_EMBEDDING={"device": "cpu"}
        self.ENCODE_KWARGS={"normalize_embeddings": True}
        # Create a single Ranker instance properly
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        self.compressor = FlashrankRerank(
            client=ranker
        )
        self.cassandraInterface=cassandra_intra
        #self.cassandraInterface.clear_tables()
        self.hf_embedding = None
        
        self.setup_vector_store()

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("LANGSMITH_API_KEY"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        self.groundedness_check = UpstageGroundednessCheck()
        self.semantic_chunker = SemanticChunker (
        self.hf_embedding, 
        )
        self.semantic_chunker.split_documents
        self.documents,self.docs_ids= [],[]
        self.parent_store = InMemoryStore()

        self.compression_retriever=  None
        self.combine_documents_chain=None
        self.memory = ConversationSummaryMemory(llm=self.llm,memory_key="chat_history",return_messages=True)

        self.chain=None
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    def cache_answer(self, question, answer):
        current_time = time.time()
        self.cache[question] = (answer, current_time)
        self.cleanup_cache()
    def cleanup_cache(self):
        current_time = time.time()
        keys_to_delete = [key for key, (_, timestamp) in self.cache.items() if current_time - timestamp >= self.cache_ttl]
        for key in keys_to_delete:
            del self.cache[key]
    def evaluate(self, html_output,css_output):
        prompt = """
Evaluate and enhance the following HTML and CSS for correctness, completeness, and UI improvements.  
Ensure the updated HTML follows these criteria:  

1. **Valid HTML Syntax**: Proper structure, closing tags, and attribute usage.  
2. **Essential Elements**: Only include content inside `<body>`, excluding `<html>` and `<head>`.  
3. **Proper Nesting**: Maintain correct hierarchy without breaking semantics.  
4. **Semantic HTML**: Use appropriate tags for accessibility and maintainability.  
5. **CSS Optimization**: Remove redundancy, improve responsiveness, and enhance UI.  
6. **Ensure styles and scripts are included inside `<style>` and `<script>` within `<body>`.**  

**HTML to evaluate:**  
{html_output}  

**CSS Output:**  
{css}  

Generate an improved **HTML structure only** that fits within a container with:  
- **max-width: 600px**, centered using `margin: 0 auto`  
- **Padding, light gray background (`#f9f9f9`), rounded corners, and a subtle shadow**  

### **Output Format:**  
Return **only** the improved HTML. Do not include explanations or additional text.
"""



        QA_CHAIN_PROMPT = PromptTemplate(template=prompt, input_variables=["html_output","css"])
        llm_chain = LLMChain(
            llm=self.llm, 
            prompt=QA_CHAIN_PROMPT, 
            callbacks=None, 
            verbose=True
        )
        evaluation = llm_chain.invoke({"html_output": html_output,"css":css_output})
        text=evaluation['text']
        text=text.replace("```","")
        text=text.replace("html","")
        text=text.replace("html\n","")

        return text
    async def setup_ensemble_retrievers(self):
        """
        Sets up an ensemble of retrievers for enhanced document retrieval.
        This method configures multiple retrievers and combines them using ensemble and compression techniques:
        - BM25 retriever for keyword-based search
        - Parent-child retriever for hierarchical document structure
        - AstraDB retriever with MMR search
        - Multi-query retriever using Gemini LLM
        - Final ensemble combining different retrieval approaches with compression
        Args:
            compressor: A compressor instance used for contextual compression of retrieved documents
        Returns:
            None - Sets up instance attributes for various retrievers
        Instance Attributes Set:
            bm25_retriever: BM25-based retriever instance
            parent_retriever: Parent-child document structure retriever
            retrieval: AstraDB retriever with MMR search
            ensemble_retriever_new: First level ensemble combining parent and AstraDB retrievers
            multi_retriever: Multi-query retriever using the first ensemble
            ensemble_retriever: Final ensemble combining BM25 and multi-query retrievers
            compression_retriever: Compressed version of the final ensemble retriever
        """
        
        parent_retriever=self.configure_parent_child_splitters()
        await ( self.add_documents_to_parent_retriever(parent_retriever))
        retrieval=self.astra_db_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'fetch_k': 50}
        )
        
        ensemble_retriever_new = EnsembleRetriever(retrievers=[parent_retriever, retrieval],
                                        weights=[0.4, 0.6])
        multi_retriever = MultiQueryRetriever.from_llm(
            ensemble_retriever_new
         , llm=self.llm
        )
        

        
        ensemble_retriever = EnsembleRetriever(retrievers=[ensemble_retriever_new, multi_retriever],
                                        weights=[0.4, 0.6])
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=self.compressor,
        base_retriever=ensemble_retriever
        )
        return compression_retriever

    async def add_documents_to_parent_retriever(self, parent_retriever:ParentDocumentRetriever):
        parent_retriever.docstore.mset(list(zip(self.docs_ids, self.documents)))
        for i,doc in enumerate(self.documents):
            doc.metadata["doc_id"] =self.docs_ids[i]

        await parent_retriever.vectorstore.aadd_documents(
        documents=self.documents,
        )
    def get_cached_answer(self, question):
        current_time = time.time()
        if question in self.cache:
            answer, timestamp = self.cache[question]
            if current_time - timestamp < self.cache_ttl:
                return answer
        return None
    def setup_vector_store(self) -> None:
        """
        Sets up the vector store for document storage and retrieval.

        This method initializes either an AstraDB vector store or falls back to a Cassandra vector store
        if AstraDB initialization fails. It handles the configuration of the storage backend used for
        storing document embeddings.

        Primary configuration:
        - Attempts to initialize AstraDBVectorStore with provided environment variables
        - Clears existing vectors in the store upon initialization

        Fallback configuration (on failure):
        - Initializes HuggingFace embeddings with 'all-MiniLM-L6-v2' model
        - Sets up Cassandra vector store as backup storage solution
        - Clears existing tables before initialization

        Raises:
            Exception: Prints error message if AstraDBVectorStore initialization fails

        Returns:
            None
        """
        try:
            self.hf_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")


            self.astra_db_store = AstraDBVectorStore(
            collection_name="langchain_unstructured",
            embedding=self.hf_embedding,
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT")
            )
            
        except Exception as e:
            print(f"Error initializing AstraDBVectorStore: {e}")

            self.hf_embedding=HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
           
            #self.CassandraInterface.clear_tables()
            self.astra_db_store :Cassandra = Cassandra(embedding=self.hf_embedding, session=self.cassandraInterface.session, keyspace=self.cassandraInterface.KEYSPACE, table_name="vectores_new")
            
            self.astra_db_store.clear()
           
    def simple_chain(self):
       

        self.prompt = """
    You are Mohamed Aziz Werghi, a skilled In cassandra database. Your task is to answer questions based on the provided context, ensuring that responses are **accurate, well-structured, and visually appealing**. 

    ### Response Guidelines:
    return list of 5 different  Expected Answers  to this question  in json format 

    #### Role: Mohamed Aziz Werghi 🤖  
    **Context:**  
    {context}  

    **Chat History:**  
    {chat_history}  

    **Question:**  
    {question}  

    """


        chain = (
            {
                "context": self.compression_retriever,
                "chat_history": lambda _: "\n".join(
                    [msg.content for msg in self.memory.load_memory_variables({}).get("chat_history", [])]
                ) if self.memory.load_memory_variables({}).get("chat_history") else "",  # Handle empty history
                "question": RunnablePassthrough()
            }
            | PromptTemplate.from_template(self.prompt)
            | self.llm
            | StrOutputParser() | parser
        )

        return chain
    def retrieval_chain(self,user:User):
        """Creates and returns a retrieval chain for question answering.

        The chain combines a compression retriever, prompt template, and Gemini LLM model to:
        1. Retrieve relevant context using compression retriever
        2. Format a prompt with the context and user question
        3. Generate an answer using the Gemini LLM
        4. Parse the output to a string

        Returns:
            Runnable: A composed chain that accepts a question as input and returns a string answer
            based only on the retrieved context. Returns "I don't know" if unable to answer.

        Example:
            chain = agent.retrieval_chain()
            answer = chain.invoke("What is X?")"""
     
        name=user.email
        job_user=user.his_job
        part_prompt = f"""
        You are a professional assistant providing expert guidance tailored to the user's role.
        Focus solely on directly answering the user's question without additional tasks.
        
        User: {name}
        Role: {job_user}
        """
        self.prompt = part_prompt + """
     

    ### Response Guidelines:
    - Format responses using **HTML** for clear presentation.
    - Use **CSS styles** to enhance readability (e.g., fonts, colors, spacing).
    - Use headings (`<h2>`, `<h3>`), lists (`<ul>`, `<ol>`), and tables (`<table>`) where appropriate.
    - Ensure code snippets are wrapped in `<pre><code>` blocks for proper formatting.
    - If styling is necessary, include minimal **inline CSS** or suggest appropriate classes.
    -Use  JavaScript is required, include <script>...</script> tags with your code for animation and manipulate dom.

    **Context:**  
    {context}  

    **Chat History:**  
    {chat_history}  

    **Question:**  
    {question}  

    **Your answer (in well-structured HTML & CSS && js  format):**
    """


        chain = (
            {
                "context": self.compression_retriever,
                "chat_history": lambda _: "\n".join(
                    [msg.content for msg in self.memory.load_memory_variables({}).get("chat_history", [])]
                ) if self.memory.load_memory_variables({}).get("chat_history") else "",  # Handle empty history
                "question": RunnablePassthrough()
            }
            | PromptTemplate.from_template(self.prompt)
            | self.llm
            | StrOutputParser()
        )

        return chain
    def configure_parent_child_splitters(self):
        """
        Configures and returns a parent-child document retrieval system using token-based text splitters.

        This method sets up a hierarchical document retrieval system with:
        - A parent splitter that creates large chunks (512 tokens)
        - A child splitter that creates smaller chunks (128 tokens) 
        - Links these splitters to vector and document stores

        Returns:
            ParentDocumentRetriever: A configured retriever that can fetch both parent and child documents,
                using the specified splitting strategy and connected to the instance's storage systems.
        """
        parent_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
        child_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)
        parent_retriever = ParentDocumentRetriever(
        vectorstore=self.astra_db_store,
        docstore=self.parent_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        )
        return parent_retriever


    
    def answer_question(self,question:str,user,request,session_id,suggestions=None):
        """
        Process a user question and return an answer using either cached responses or generating a new one.

        This method first checks if an identical question has been answered before. If found, returns
        the cached response. Otherwise, processes the question through a retrieval chain to generate
        a new answer, stores it, and returns the result.

        Parameters:
            question (str): The user's question to be answered
            request: The original request object 
            session_id: Unique identifier for the user session

        Returns:
            dict: A dictionary containing:
                - request: The original request object
                - answer (str): The response to the question
                - question (str): The original question
                - partition_id (str, optional): ID of the cached response partition if exists
                - timestamp (datetime, optional): Timestamp of the cached response if exists
                - evaluation (bool): Always True, indicates response evaluation status

        Raises:
            Any exceptions from underlying services (CassandraInterface, retrieval_chain) are not explicitly handled
        """
        self.logger.info(f"Received question: {question}")
        exist_answer=self.get_cached_answer(question)
        if exist_answer is not None:
            return self.generate_message_html(question, exist_answer)
        else:
            question_enhanced= question + "suggestions:words u may need to use "+"\n".join(suggestions)
            final_answer=None

            try:
                context=self.compression_retriever.invoke(question_enhanced)
                context_memory= self.memory.load_memory_variables({}).get("chat_history", [])
                if context_memory:
                    context_memory = "\n".join([msg.content for msg in context_memory])
                    context = f"{context}\n{context_memory}"
                
                final_answer = self.chain.invoke(question_enhanced)
                PROMPT = """
                Generate CSS from the given HTML.
                HTML: {Html}
                
                CSS: [css]
                """
                PROMPT_out= PROMPT.format(
                    Html=final_answer
                )
                css_reponse=self.llm.invoke(PROMPT_out)
                
                refined_answer = self.evaluate(final_answer,css_reponse)
                final_answer=refined_answer
                self.logger.info(f"Answer provided: {final_answer}")
                
               
            except Exception as e:
                self.logger.error(f"Error while answering question: {e}")
                self.chain=self.retrieval_chain(self.llm)
                final_answer = self.chain.invoke(question_enhanced)
            #final_answer= self.answer_rewriting(f"{final_answer}",question)
            self.cache_answer(question, final_answer)
            self.memory.save_context({"question": question}, {"answer": f"{final_answer}"})
           
            self.cassandraInterface.insert_answer(session_id,user,question,final_answer)
            reponse= self.generate_message_html(question, final_answer)
         
            return reponse
    def make_agent(self, question: str, tools: list):
        
        planner = load_chat_planner(self.llm)
        
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        agent = PlanAndExecute(planner=planner, executor=executor)

        # Ask the agent
        answer = agent.invoke({"input": question})

        return answer
    def generate_message_html(self, question, final_answer):
        message_html =generate_answer_html(question, final_answer,False)

        return HTMLResponse(content=message_html, status_code=200)
#https://medium.com/@irmjoshy/multi-agent-models-for-webpage-development-leveraging-ai-for-code-generation-and-verification-52cb60cec9eb
#https://github.com/microsoft/generative-ai-for-beginners/tree/main/03-using-generative-ai-responsibly
# agent=AgentInterface()

# @tool
# def write_cql(question: str):
#     """
#     Convert a natural language question into a CQL query, execute it on the database,
#     and use an LLM to answer the question based on the query results.

#     Args:
#         question (str): A user question about the data (e.g., "List all users older than 25").

#     Returns:
#         str: A human-readable answer generated by the LLM using the data retrieved via the CQL query.
#     """
#     cql_agent = CQLAGENT()
#     return cql_agent.answer_data_with_cql(question)

# @tool
# def basic_scraper(url: str) -> str:
#     """
#     Scrape the webpage at the given URL and return the text content.
#     This function uses the requests and BeautifulSoup libraries to fetch and parse the HTML content of the webpage.
#     Args:
#         url (str): The URL of the webpage to scrape.
#     Returns:
#         str: The scraped text from the webpage.
    
#     """
#     import requests
#     from bs4 import BeautifulSoup
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     return soup.get_text()[:1000]  # Limit to avoid overload
# tools = [TavilySearchResults(), basic_scraper,write_cql,
         
#          ]

# Answer = agent.make_agent(
#     "Help me learn how to write code in the Crystal programming language. You can use tutorials from the internet, and you might need to scrape data if needed.",
#     tools=tools
# )

# print(Answer['output'])