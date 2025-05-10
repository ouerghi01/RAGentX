from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
import os
from dotenv import load_dotenv
load_dotenv()


def Sql_agent(llm):
    db_uri = os.getenv("POSTGRES_DB_URI")
    if db_uri is None:
        raise ValueError("DATABASE_URL environment variable not set.")
    db = SQLDatabase.from_uri(db_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
    return agent

#agent = Sql_agent(llm)

# Query in natural language
#response = agent.run("List all products purchased by Bob Smith and their quantities.")
