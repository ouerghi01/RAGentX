from langchain_community.document_loaders import CassandraLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.storage import InMemoryStore
from langchain.text_splitter import TokenTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_core.documents import Document
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os 
from pathlib import Path
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_community.document_loaders import (
    unstructured,
    
)


from services.pdf_service import PDFService
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
load_dotenv()  # take environment variables from .env.
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY")
from langchain_community.document_loaders import PDFPlumberLoader  
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

from langchain_upstage import ChatUpstage,UpstageGroundednessCheck
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
    def __init__(self,role):
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
        self.prompt=None
        self.UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR"))  
        self.MODEL_NAME_llm=os.getenv("MODEL_NAME")
        self.BASE_URL_OLLAMA=os.getenv("OLLAMA_BASE_URL")
        self.MODEL_NAME_EMBEDDING=os.getenv("MODEL_NAME_EMBEDDING")
        self.MODEL_KWARGS_EMBEDDING={"device": "cpu"}
        self.ENCODE_KWARGS={"normalize_embeddings": True}
        self.pdfservice=PDFService(self.UPLOAD_DIR)
        Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=3)
        self.CassandraInterface=CassandraInterface()
        self.hf_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
        
        self.semantic_chunker = SemanticChunker (
        self.hf_embedding, 
        )
        self.setup_vector_store()
            
        self.llm_gemini = ChatGoogleGenerativeAI(
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
        self.chat_up=ChatUpstage()

        self.groundedness_check = UpstageGroundednessCheck()

        self.documents=self.load_pdf_documents_fast()
        self.parent_store = InMemoryStore()

        self.compression_retriever=self.setup_ensemble_retrievers(compressor)
        #https://aistudio.google.com/app/u/2/apikey
        
        self.combine_documents_chain=None

        self.llm=Ollama(model=self.MODEL_NAME_llm, base_url=self.BASE_URL_OLLAMA,verbose=True,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),)
        self.memory = ConversationSummaryMemory(llm=self.llm_gemini,memory_key="chat_history",return_messages=True)
        
        self.chain=self.retrieval_chain(self.llm_gemini)

    def setup_ensemble_retrievers(self, compressor):
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
        bm25_retriever=BM25Retriever.from_documents(self.documents)
        bm25_retriever.k=2 
        parent_retriever=self.configure_parent_child_splitters()
        
        parent_retriever.add_documents(self.documents)

        retrieval=self.astra_db_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'fetch_k': 50}
        )
        ensemble_retriever_new = EnsembleRetriever(retrievers=[parent_retriever, retrieval],
                                        weights=[0.4, 0.6])
        multi_retriever = MultiQueryRetriever.from_llm(
            ensemble_retriever_new
         , llm=self.llm_gemini
        )
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, multi_retriever],
                                        weights=[0.4, 0.6])
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
        )
        return compression_retriever

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
            self.astra_db_store = AstraDBVectorStore(
            collection_name="langchain_unstructured",
            embedding=self.hf_embedding,
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT")
            )
            self.astra_db_store.clear()
        except Exception as e:
            print(f"Error initializing AstraDBVectorStore: {e}")
            self.hf_embedding=HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.CassandraInterface.clear_tables()
            self.astra_db_store :Cassandra = Cassandra(embedding=self.hf_embedding, session=self.session, keyspace=CassandraInterface().KEYSPACE, table_name="vectores_new")
        

    def retrieval_chain(self, llm=None):
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
        
        self.prompt = """
        u are an intelligent agent  designed to answer questions based solely on the information provided in the context. Your goal is to provide accurate, clear, and concise answers while respecting the boundaries of the available context.
        Answer the question based only on the supplied context. If you don't know the answer, say "I don't know".
        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        Your answer:
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
            | self.chat_up
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
        """Builds a question answering process using a language model and a retriever.
        This method initializes and configures a question answering pipeline that leverages a language model (either `self.llm_gemini` or `self.llm`) and a retriever (`self.compression_retriever`) to answer questions based on a provided context.
        The process involves the following steps:
        1.  **Define a prompt:** A prompt is defined to guide the language model in generating answers based on the context and question. The prompt instructs the model to:
            *   Answer questions using only the provided context.
            *   Respond with "Je ne sais pas." if the answer is not in the context.
            *   Provide concise answers (maximum three sentences).
            *   Explain the usage and purpose of tools, functionalities, or concepts mentioned in the context.
            *   Avoid making assumptions or extrapolating information.
            *   Provide relevant suggestions based on the context, if possible.
        2.  **Create an LLMChain:** An `LLMChain` is created using the language model and the defined prompt. This chain is responsible for generating answers based on the context and question. If `self.llm_gemini` fails, it falls back to `self.llm`.
        3.  **Define a document prompt:** A document prompt is defined to format the input documents for the language model. This prompt includes the page content and the source of the document.
        4.  **Create a StuffDocumentsChain:** A `StuffDocumentsChain` is created to combine the documents retrieved by the retriever into a single context for the language model. This chain uses the `LLMChain` and the document prompt to format the documents and pass them to the language model.
        5.  **Create a RetrievalQA object:** A `RetrievalQA` object is created to combine the retriever and the document chain into a single question answering pipeline. This object takes a question as input, retrieves relevant documents using the retriever, combines the documents into a context using the document chain, and generates an answer using the language model.
        Returns:
            RetrievalQA: A `RetrievalQA` object that can be used to answer questions based on the provided context.

       """
   
        prompt = """
        Je suis un agent intelligent conçu pour répondre aux questions en m’appuyant exclusivement sur les informations contenues dans la documentation fournie. Mon objectif est de fournir des réponses précises, claires et concises, tout en respectant les limites du contexte disponible.  

        📌 **Directives pour formuler les réponses :**  
        1. **Utilisation stricte du contexte** : Je dois répondre uniquement en me basant sur les informations du contexte fourni.  
        2. **Gestion des inconnues** : Si la réponse n’est pas présente dans la documentation, je dois répondre : *"Je ne sais pas."*  
        3. **Clarté et concision** : Les réponses doivent être courtes (maximum trois phrases), sans ajouter d’informations non mentionnées dans le contexte.  
        4. **Explication des fonctionnalités** : Si le contexte mentionne un outil, une fonctionnalité ou un concept spécifique, je dois expliquer son utilisation et son objectif.  
        5. **Aucune supposition** : Je ne dois jamais inventer ou extrapoler des informations non fournies.  
        6. **Recommandation pertinente** : Si possible, je peux donner une suggestion utile basée sur le contexte.  

        📖 **Contexte** :  
        {context}  

        ❓ **Question** :  
        {question}  

        ✍️ **Réponse** :

        """

        QA_CHAIN_PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question"])

        try:
            llm_chain = LLMChain(
            llm=self.llm_gemini, 
            prompt=QA_CHAIN_PROMPT, 
            callbacks=None, 
            verbose=True
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            llm_chain = LLMChain(
            llm=self.llm, 
            prompt=QA_CHAIN_PROMPT, 
            callbacks=None, 
            verbose=True
            )

        document_prompt = PromptTemplate(
            template="Context:\ncontent:{page_content}\nsource:{source}",  
            input_variables=["page_content", "source"]  
        )
        
        self.combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,  
            document_variable_name="context",
            callbacks=None,
            document_prompt=document_prompt  
        )
        qa = RetrievalQA(
            combine_documents_chain=self.combine_documents_chain,  
            retriever=self.compression_retriever,
            verbose=True
        )
        return qa
    # best way to handle conversation history 
    def build_retrieval_history(self):
        """Builds a conversational retrieval chain for question answering.

            This method initializes and configures a ConversationalRetrievalChain using
            the specified language model, retriever, and memory. The chain is designed
            to handle conversational question answering, leveraging the retriever to
            fetch relevant documents and the memory to maintain conversation history.

            Returns:
                ConversationalRetrievalChain: A configured conversational retrieval chain.
            """
        qa = ConversationalRetrievalChain.from_llm(self.llm_gemini, retriever=self.compression_retriever, memory=self.memory)
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
    def load_pdf_documents_fast(self,bool=True):
        """
        Loads and processes PDF documents from a specified directory with semantic chunking.
        This method reads all PDF files from the UPLOAD_DIR directory, processes them using
        PDFPlumberLoader, and splits the documents into semantic chunks.
        Parameters:
            bool (bool): Flag to control document loading. If True, loads and processes documents.
                        Defaults to True.
        Returns:
            list: A list of processed document chunks. Returns empty list if bool is False.
        Note:
            - Creates UPLOAD_DIR if it doesn't exist
            - Only processes files with '.pdf' extension
            - Applies semantic chunking to each loaded document
            - Skips any None documents during processing
        """
        documents = []
        if bool==True:
            self.UPLOAD_DIR.mkdir(exist_ok=True) 
            docs=[]
            for file in os.listdir(self.UPLOAD_DIR):
                if file.endswith(".pdf"):
                    
                        full_path = self.UPLOAD_DIR/ file
                        loader = PDFPlumberLoader(full_path)
                        document = loader.load()
                        docs.append(document)
            for doc in docs:
                if doc is not None:
                    chunks = self.semantic_chunker.split_documents(doc)
                    documents.extend(chunks)  
            
        #self.load_documents_from_cassandra(documents)
                
        return documents
    def load_pdf_v2(self):
        documents = []
        for filename in os.listdir(self.UPLOAD_DIR):
            file_path = self.UPLOAD_DIR / filename
            if not file_path.is_file():
                continue
            elements = unstructured.get_elements_from_api(
                file_path=file_path,
                api_key=os.getenv("UNSTRUCTURED_API_KEY"),
                api_url=os.getenv("UNSTRUCTURED_API_URL"),
                strategy="fast", # default "auto"
                pdf_infer_table_structure=True,
            )

            documents.extend(self.process_elements_to_documents(elements))
        return documents

    def process_elements_to_documents(self,elements):
        documents = []
        current_doc = None

        for el in elements:
            if el.category in ["Header", "Footer"]:
                continue # skip these
            if el.category == "Title":
                if current_doc is not None:
                    documents.append(current_doc)
                current_doc = None
            if not current_doc:
                current_doc = Document(page_content="", metadata=el.metadata.to_dict())
            current_doc.page_content += el.metadata.text_as_html if el.category == "Table" else el.text
            if el.category == "Table":
                if current_doc is not None:
                    documents.append(current_doc)
                current_doc = None
        return documents
    def load_pdf_documents_with_ocr(self, bool=True):
        """
        Load PDF documents and convert them to text format, including both regular text and text extracted from images.
        This method processes PDF documents stored in the upload directory, combining the regular text content
        with text extracted from images found in the PDFs. Each document is converted into a Document object
        with associated metadata.
        Args:
            bool (bool, optional): Flag to control whether documents should be loaded. Defaults to True.
        Returns:
            list: A list of Document objects, where each Document contains:
                - page_content: Combined text from both regular content and image-extracted text
                - metadata: Associated metadata dictionary for the document
        Note:
            - Creates upload directory if it doesn't exist
            - Uses PDFService to extract data from PDF files
            - Combines regular text and image-extracted text with newline separators
            - Validates metadata is in dictionary format
        """
        documents = []
        if bool==True:
            self.UPLOAD_DIR.mkdir(exist_ok=True) 
            records=self.pdfservice.extract_data_from_pdf_directory()
            docs=[]
            for record in records:
                
                full_text=record['text'] + "\n" +  "\n" + record['images_to_text']
                meta = record['meta'] if isinstance(record['meta'], dict) else {}
                doc = Document(page_content=full_text, metadata=meta)
                docs.append(doc)

            documents.extend(docs)
            
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
        exist_answer=self.CassandraInterface.find_same_reponse(question)
        if exist_answer is not None:
            return{"request": request, "answer": exist_answer.answer, "question": question,"partition_id":exist_answer.partition_id,"timestamp":exist_answer.timestamp,"evaluation":True}
        else:
            question_enhanced= question 
            final_answer=None
            #chat_history = self.CassandraInterface.get_chat_history(session_id)
            try:
                context=self.compression_retriever.invoke(question_enhanced)
                context_memory= self.memory.load_memory_variables({}).get("chat_history", [])
                if context_memory:
                    context_memory = "\n".join([msg.content for msg in context_memory])
                    context = f"{context}\n{context_memory}"
                final_answer = self.chain.invoke(question_enhanced)
                gc_result=self.groundedness_check.invoke({
                "context": context,
                "answer": final_answer,
                })
                if gc_result.lower().startswith("grounded"):
                    print("Grounded")
                else:
                    print("Not Grounded")
            except Exception as e:
                print(f"End usage of api gemini {e}")
                self.chain=self.retrieval_chain(self.llm)
                final_answer = self.chain.invoke(question_enhanced)
            final_answer= self.answer_rewriting(f"{final_answer}",question)
            self.memory.save_context({"question": question}, {"answer": f"{final_answer}"})

            self.CassandraInterface.insert_answer(session_id,question,final_answer)
            return {"request": request, "answer": final_answer, "question": question,"evaluation":True}

