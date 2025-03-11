from langchain_community.document_loaders import CassandraLoader
from langchain.retrievers import ContextualCompressionRetriever
import logging
import time
from collections import OrderedDict
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor
import threading
from fastapi.responses import HTMLResponse
import uuid
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
from dotenv import load_dotenv
import os 
from pathlib import Path
from langchain.retrievers import BM25Retriever, EnsembleRetriever


from langchain_community.document_loaders import (
    unstructured,
    
)


from services.pdf_service import PDFService
from langchain_core.prompts import ChatPromptTemplate
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
