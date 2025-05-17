import os

import threading
from services.fastapp import FastApp

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import asyncio

from services.consumer   import PowerBIKafkaConsumer
##https://github.com/UpstageAI/cookbook/blob/main/Solar-Fullstack-LLM-101/10_tool_RAG.ipynb
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
if __name__ == "__main__":
     app=FastApp()
     consumer = PowerBIKafkaConsumer(
        topic="cassandrastream",
        group_id="cassandrastream-group",
        
        bootstrap_servers='localhost:9092',
        power_bi_url=os.getenv("POWER_BI_URL")
    )
     app_thread = threading.Thread(target=app.run)
     consumer_thread = threading.Thread(target=consumer.consume_loop)
     
     app_thread.start()
     consumer_thread.start()
     app_thread.join()
     consumer_thread.join()
# https://github.com/bhattbhavesh91/pdf-qa-astradb-langchain/blob/main/requirements.txt
# https://github.com/michelderu/chat-with-your-data-in-cassandra/blob/main/docker-compose.yml
#https://medium.com/@o39joey/advanced-rag-with-python-langchain-8c3528ed9ff5
#https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html
#https://github.com/langchain-ai/langchain/discussions/9158*
#https://github.com/michelderu/build-your-own-rag-chatbot/blob/main/app_6.py








