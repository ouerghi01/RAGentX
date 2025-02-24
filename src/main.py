import uvicorn
from services.agent_service import AgentInterface
from fastapi import FastAPI 
from fastapi import FastAPI,Request,File,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

class FastApp :
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
        print("Starting App")
        self.agent=AgentInterface("assistant")
        self.session_id=self.agent.CassandraInterface.create_room_session()
        self.templates = Jinja2Templates(directory="templates")
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
    def send_message(self,request:Request,question:str=Form(...)):
        
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