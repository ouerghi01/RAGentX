import ctypes
from google import genai

import uvicorn
from services.trie_service import TrieService
from services.agent_service import AgentInterface
from dotenv import load_dotenv
import string 
import pandas as pd 
from services.Crawler import main_crawler
load_dotenv()  # take environment variables from .env.
from fastapi import FastAPI,Request,Form, WebSocket 
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from services.auth_service import AuthService
from services.cassandra_service import CassandraManager
from fastapi import  FastAPI
import random
##https://github.com/UpstageAI/cookbook/blob/main/Solar-Fullstack-LLM-101/10_tool_RAG.ipynb
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
from fastapi import Request, HTTPException
class PredictNextWord:
     def __init__(self):
          self.prompt="""
            Your role is to complete or correct the following sentence:  
            {sentence}  
            {suggestions}
            Provide the full corrected or completed sentence.
            Provide different suggestions separated by a comma.
            
            """
          self.cache={}
          translator = str.maketrans('', '', string.punctuation)
          df = pd.read_csv("Data_with_explanations.csv")
          if "question" not in df.columns:
                raise ValueError("Column 'question' not found in CSV!")
          questions = df["question"].dropna().tolist()
          data = [word.translate(translator).lower() for sentence in questions for word in sentence.split() if word not in string.punctuation]
          tokens = (ctypes.c_char_p * len(data))(*(word.encode("utf-8") for word in data))
          self.treeServ= TrieService(tokens=tokens)
          self.llm=genai.Client(api_key="AIzaSyAcIGFo53M8vf2eb_UO4JGBYb0an7B8xH4").chats.create(model="gemini-2.0-flash")
     async def predict_next_word(self,websocket: WebSocket):
        await websocket.accept()
        while True:

            data = await websocket.receive_text()
            if data in self.cache:
                await websocket.send_text(f" {self.cache[data]}")
            else:
                token_last_word = data.split()[-1].lower()

                suggestions = self.treeServ.autocomplete(token_last_word)
                completion = self.llm.send_message(self.prompt.format(sentence=data,suggestions=suggestions)).text
                self.cache[data]=completion
                await websocket.send_text(f" {completion}")
class FastApp :
    """FastAPI Application wrapper for RAG-based chatbot implementation.
    This class sets up a FastAPI application with CORS middleware, routes, and event handlers
    for a chatbot interface using RAG (Retrieval Augmented Generation).
    Attributes:
        app (FastAPI): The FastAPI application instance
        agent (AgentInterface): Agent handling chat interactions and RAG functionality
        templates (Jinja2Templates): Jinja2 templates for HTML rendering
        origins (list): Allowed origins for CORS
        session_id (str): Unique identifier for the current chat session
    Methods:
        startup_event(): Initializes agent, creates session and sets up templates on app startup
        send_message(request, question): Handles incoming chat messages and returns responses
        shutdown_event(): Cleans up resources on app shutdown
        main_page(request): Renders the main chat interface
        run(): Starts the FastAPI server
    Routes:
        / : Main chat interface (GET)
        /send_message/ : Message handling endpoint (POST)
    """
    def __init__(self):
        self.cassandra_intra=CassandraManager()
        self.app=FastAPI()
        self.agent=None
        self.auth_service=AuthService(self.cassandra_intra,self.agent)
        self.templates = None
        self.agent_word_predictor=PredictNextWord()
        self.origins =  [
        "http://192.168.100.66:3000/",
        "http://localhost:3000/",
        ]
        self.app.add_middleware(
        CORSMiddleware,
        allow_origins=self.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        )
        self.session_id=None
        
        self.app.add_event_handler("startup", self.startup_event)
        self.app.add_event_handler("shutdown", self.shutdown_event)
        self.app.add_api_route("/", self.main_page)
        self.app.add_api_route("/send_message/", self.send_message, methods=["POST"])
        self.app.add_api_route("/register/", self.auth_service.register_user, methods=["POST"])
        self.app.add_api_route("/login/", self.auth_service.login_for_access_token, methods=["POST"])
        self.app.add_api_route("/uploads/", self.upload_file, methods=["POST"])
        self.app.add_api_route("/logout/",self.auth_service.logout,methods=["POST"])
        self.app.add_api_route("/verify/",self.verify_jwt,methods=["POST"])
        #self.app.add_middleware(self.auth_service.verify_token)
        self.app.add_websocket_route("/ws", self.agent_word_predictor.predict_next_word)
    
    async def upload_file(self,request: Request):
        jwt=request.cookies.get("auth_token")
        current_user=await self.auth_service.get_current_user(jwt)
        if( current_user is None):
                raise HTTPException(status_code=401, detail="Authentication failed: No valid token provided.")
        
        file=await request.body()
        id=random.randint(1,100000)
        file_name="uploads/"+str(id)+".pdf"
        with open(file_name, "wb") as f:
            f.write(file)
        # this is a dump implementation, you should implement a better way to handle the file
        # like uploading to a cloud storage and then saving the link in the database
        # or saving the file in a database
        # or saving the file in a directory and then saving the link in the database
        self.agent=AgentInterface("assistant",self.cassandra_intra)
        self.agent.compression_retriever=await self.agent.setup_ensemble_retrievers()
        self.agent.chain=self.agent.retrieval_chain(current_user)
        return {
            "message":"File uploaded successfully"
        }
    
    async def startup_event(self):
        """
        Initializes the application by setting up necessary components.

        This method performs the following initialization tasks:
        1. Creates an agent interface instance
        2. Generates a new room session ID
        3. Sets up Jinja2 templates directory
        4. Mounts static files directory

        Returns:
            None
        """
        print("Starting App ...")
        #contents,urls = await main_crawler()
        self.agent=AgentInterface("assistant",self.cassandra_intra,name_dir="/home/aziz/IA-DeepSeek-RAG-IMPL/APP/uploads")
        self.agent.compression_retriever=await self.agent.setup_ensemble_retrievers()
        self.agent.chain=None
        self.auth_service.agent=self.agent
        self.session_id=self.cassandra_intra.create_room_session()
        self.templates = Jinja2Templates(directory="templates")
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
    async def send_message(self,request:Request,question:str=Form(...)):
        """
        Handles sending messages and retrieving responses from an agent.
        This method processes a message request, gets an answer from the agent, and returns
        a templated response.
        Args:
            request (Request): The FastAPI request object.
            question (str): The question text submitted via form data.
        Returns:
            TemplateResponse: A templated response containing the agent's answer and related data.
        Example:
            response = send_message(request, "What is the weather today?")
        """
        jwt=request.cookies.get("auth_token")
        current_user=await self.auth_service.get_current_user(jwt)
        if( current_user is None):
                raise HTTPException(status_code=401, detail="Authentication failed: No valid token provided.")

        html_answer= self.agent.answer_question(question,request,self.session_id)
        return html_answer
    
    def shutdown_event(self):
        self.agent.CassandraInterface.session.shutdown()
        self.agent=None
        self.templates = None
    async def verify_jwt(self,request:Request,jwt:str=Form(...)):
        current_user=await self.auth_service.get_current_user(jwt)
        if( current_user is None):
                raise HTTPException(status_code=401, detail="Authentication failed: No valid token provided.")
        return current_user
    async def main_page(self,request: Request):
        jwt=request.cookies.get("auth_token")
        current_user=await self.auth_service.get_current_user(jwt)
        if( current_user is None):
                self.agent.chain=None
        else:
             
            self.agent.chain=self.agent.retrieval_chain(current_user)
        return self.templates.TemplateResponse("index.html", {"request": request})
    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
     app=FastApp()
     app.run()
# https://github.com/bhattbhavesh91/pdf-qa-astradb-langchain/blob/main/requirements.txt
# https://github.com/michelderu/chat-with-your-data-in-cassandra/blob/main/docker-compose.yml
#https://medium.com/@o39joey/advanced-rag-with-python-langchain-8c3528ed9ff5
#https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html
#https://github.com/langchain-ai/langchain/discussions/9158*
#https://github.com/michelderu/build-your-own-rag-chatbot/blob/main/app_6.py








