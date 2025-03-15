import uvicorn
from services.agent_service import AgentInterface
from dotenv import load_dotenv

from services.Crawler import main_crawler
load_dotenv()  # take environment variables from .env.
from fastapi import FastAPI,Request,Form 
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from services.auth_service import AuthService
from services.cassandra_service import CassandraInterface
from fastapi import  FastAPI
import random
##https://github.com/UpstageAI/cookbook/blob/main/Solar-Fullstack-LLM-101/10_tool_RAG.ipynb
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

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
        self.cassandra_intra=CassandraInterface()
        self.app=FastAPI()
        self.auth_service=AuthService(self.cassandra_intra)
        self.agent=None
        self.templates = None
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
    async def upload_file(self,request: Request):
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
        self.agent.chain=self.agent.retrieval_chain()
        self.session_id=self.cassandra_intra.create_room_session()
        self.templates = Jinja2Templates(directory="templates")
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
    def send_message(self,request:Request,question:str=Form(...)):
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
        
        html_answer= self.agent.answer_question(question,request,self.session_id)
        return html_answer
    
    def shutdown_event(self):
        self.agent.CassandraInterface.session.shutdown()
        self.agent=None
        self.templates = None
    def main_page(self,request: Request):
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








