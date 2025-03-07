import uvicorn
from services.agent_service import AgentInterface
from fastapi import FastAPI 
from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
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
        self.app=FastAPI()
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

    
    def startup_event(self):
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
        self.agent=AgentInterface("assistant")
        self.session_id=self.agent.CassandraInterface.create_room_session()
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
        
        json_data= self.agent.answer_question(question,request,self.session_id)
        return self.templates.TemplateResponse(
        "messages.html",json_data
        ) 
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
#https://github.com/langchain-ai/langchain/discussions/9158