from datetime import datetime, time
import multiprocessing
from typing import Annotated
import uuid
from fastapi.responses import  JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from langchain.agents import Tool
from pydantic import BaseModel
from sse_starlette import EventSourceResponse
import uvicorn
import random

from services.PredictNextWord import PredictNextWord
from services.response_html import generate_answer_html
from services.agent_service import AgentInterface
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
from fastapi import Depends, FastAPI,Request,Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from services.auth_service import AuthService
from services.cassandra_service import CassandraManager
from fastapi import  FastAPI
##https://github.com/UpstageAI/cookbook/blob/main/Solar-Fullstack-LLM-101/10_tool_RAG.ipynb
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
from fastapi import Request, HTTPException
class CloseSessionBody(BaseModel):
    jwt: str
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
        self.sessions={}
        self.agents= {}
        
        self.app.add_event_handler("startup", self.startup_event)
        self.app.add_event_handler("shutdown", self.shutdown_event)
        self.app.add_api_route("/", self.main_page)
        self.app.add_api_route("/send_message/", self.send_message, methods=["POST"])
        self.app.add_api_route("/register/", self.auth_service.register_user, methods=["POST"])
        self.app.add_api_route("/login/", self.login, methods=["POST"])
        self.app.add_api_route("/uploads/", self.upload_file, methods=["POST"])
        self.app.add_api_route("/logout/",self.logout,methods=["POST"])
        self.app.add_api_route("/verify/",self.verify_jwt,methods=["POST"])
        self.app.add_api_route("/create_session/",self.create_session,methods=["GET"])
        self.app.add_api_route("/get_sessions",self.get_sessions,methods=["GET"])
        self.app.add_api_route("/close_session",self.close_session,methods=["POST"])
        self.app.add_api_route("/get_conversation_history/{session_id}",self.get_conversation_history,methods=["GET"])
        #self.app.add_middleware(self.auth_service.verify_token)
        self.app.add_websocket_route("/ws", self.agent_word_predictor.predict_next_word)
    from fastapi import File, UploadFile, HTTPException
    async def logout(self,request:Request,response:Response):
        user= await self.validate_auth_token(request)
        if user:
            self.cassandra_intra.close_room_session(self.sessions[user.username])
            msg=await self.auth_service.logout(request,response)
            self.sessions[user.username]=None
            response.delete_cookie("auth_token")
            return msg
        else:
            raise HTTPException(status_code=401, detail="Authentication failed: No valid token provided.")
    async def login (self,response:Response,
            form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
        ) :
        token=await self.auth_service.login_for_access_token(response,form_data)
        if (token ):
            if (self.sessions.get(token.the_user) is None):
                self.sessions[token.the_user]=self.cassandra_intra.create_room_session(token.the_user)
            asyncio.create_task(self.add_agent_to_user(token))

            return token 
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def add_agent_to_user(self, token):
        if(self.agents== {}):
            agent=AgentInterface()
        else:
            agent = random.choice(list(self.agents.values()))
        agent.compression_retriever=await agent.setup_ensemble_retrievers()
        current_user=await self.auth_service.get_current_user(token.access_token)
        agent=self.complete_agent(current_user,agent)
        self.agents[current_user.username]=agent
    async def upload_file(self,file: UploadFile = File(...)):
         
         
        contents = file.file.read()
        file_name="uploads/"+file.filename
        with open(file_name, "wb") as f:
            f.write(contents)
        
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
        
        #self.auth_service.agent=None
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
        suggestions=[]
        time_sended_question=datetime.now()
        session_id=self.sessions.get(current_user.username)
        agent : AgentInterface=self.agents.get(current_user.username)
        html_answer= agent.answer_question(question,current_user,request,session_id,suggestions,time_sended_question,
                                                )
        return html_answer
    
    def shutdown_event(self):
        self.agents={}
        self.sessions={}
        self.templates = None
    async def verify_jwt(self,request:Request,jwt:str=Form(...)):
        current_user=await self.auth_service.get_current_user(jwt)
        if( current_user is None):
                raise HTTPException(status_code=401, detail="Authentication failed: No valid token provided.")
        return current_user
    async def main_page(self,request: Request):
        jwt=request.cookies.get("auth_token")
        if jwt is None:
            return self.templates.TemplateResponse("index.html", {"request": request})
            
        current_user=await self.auth_service.get_current_user(jwt)
        
        if( current_user is None):
                self.agents={}
        else:
            session_conv=self.cassandra_intra.get_last_session_created_by_user(current_user.username)
            self.sessions[current_user.username]=session_conv.session_id
            self.add_agent_to_user(jwt)
            #self.complete_agent(current_user)
        return self.templates.TemplateResponse("index.html", {"request": request})

    def complete_agent(self, current_user,agent):
        from langchain.agents import initialize_agent
        from langchain.agents.agent_types import AgentType
        agent.chain=agent.retrieval_chain(current_user)
        agent.rag_tool=Tool(
                name="Deep Answering Agent",
                func=agent.chain.invoke,
                description = (
    "This tool is backed by a full Retrieval-Augmented Generation (RAG) agent, optimized for deep and context-aware information retrieval. "
    "It is invoked by the main agent when a query requires in-depth reasoning or highly specific knowledge that cannot be handled by simple "
    "complex and research-intensive tasks."
)
            )
        agent.tools.append(agent.rag_tool)
        agent.final_agent=initialize_agent(
                 agent.tools,
                llm=agent.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
        return agent
    async def validate_auth_token(self, jwt : Request| str):
        jwt= jwt if isinstance(jwt, str) else jwt.cookies.get("auth_token")
        if jwt is None:
            raise HTTPException(status_code=401, detail="Authentication failed: No valid token provided.")
        current_user=await self.auth_service.get_current_user(jwt)
        if( current_user is None):
                   raise HTTPException(status_code=401, detail="Authentication failed: No valid token provided.")
        return current_user
    async def get_conversation_history(self,request:Request,session_id:str):
         
         current_user=await self.validate_auth_token(request)
         if ( current_user is None or self.sessions.get(current_user.username) is None):
                raise HTTPException(status_code=401, detail="Authentication failed: No valid token provided.")
         
         self.sessions[current_user.username]=session_id
         answers_hist= self.cassandra_intra.get_chat_history(session_id)
         self.agent.memory.clear()
         self.agent.memory_llm=[]
         answers_list_html= []
         for hist in answers_hist:
                question=hist[0]
                answer=hist[1]
                if self.agent is not None:
                    self.agent.memory.save_context({"question": question}, {"answer": f"{answer}"})
                    self.agent.memory_llm.append((question, answer))
                answers_list_html.append(generate_answer_html(question,
                                                              (answer),
                                                              True))
         
         return JSONResponse(content=answers_list_html)
    async def close_session(self,request:Request):
         jwt=request.cookies.get("auth_token")
         current_user=await self.validate_auth_token(jwt)
         if ( current_user is None or self.sessions.get(current_user.username) is None):
                raise HTTPException(status_code=401, detail="Authentication failed: No valid token provided.")
         session_id=self.sessions.get(current_user.username)
         self.cassandra_intra.close_room_session(session_id)
         username=current_user.username
         self.agents[username]=self.agents[username].memory.clear()
         self.agents[username]=self.agents[username].memory_llm.clear()
         return JSONResponse(content="Session closed")
    async def create_session(self,request:Request):
         current_user=await self.validate_auth_token(request)
         
         session_id=self.cassandra_intra.create_room_session(
             current_user.username
             ) # uuid type return
         self.sessions[current_user.username]=session_id
         self.agents[current_user]=self.agents[current_user].memory.clear()
         self.agents[current_user]=self.agents[current_user].memory_llm.clear()
         
         return JSONResponse(content=str(session_id))
    async def get_sessions(self, request: Request):
        async def event_generator():
             while True:
                if await request.is_disconnected():
                    break
                jwt = request.cookies.get("auth_token")
                if jwt is None:
                    yield {
                        "event": "message",
                        "data": "<div>⚠️ Not authenticated</div>"
                    }
                    break
                user = await self.validate_auth_token(request)

                if not user:
                    yield {
                        "event": "message",
                        "data": "<div>⚠️ Not authenticated</div>"
                    }
                    break

                sessions = self.cassandra_intra.get_sessions(user)

                # Render the HTML snippet from your Jinja2 template
                html_content = self.templates.get_template("sessions.html").render(sessions=sessions)

                # Yield the HTML as a plain string
                yield {
                    "event": "message",
                    "data": html_content
                }

                await asyncio.sleep(5)

        return EventSourceResponse(event_generator())
    def run(self):
        
         
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=8000,
            workers=multiprocessing.cpu_count()  # or a fixed number like 4
        )
        server = uvicorn.Server(config)
        server.run()