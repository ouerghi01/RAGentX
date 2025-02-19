from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import *

# Initialize retriever
session = initialize_database_session()
retriever = create_retriever_from_documents(session, load_pdf_documents())

llm = Ollama(model="qwen2.5:3b", base_url="http://localhost:11434")

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "Context: {context}"),  # <-- Added missing context variable
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create the RAG chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)
query="Select question,answer from store_key.response_table ;"
result_set = session.execute(query)
chat_history = []
for row in result_set:
    chat_history.append(("human",row.question))
    chat_history.append(("assistant",row.response))


query = "Comment puis-je valider ou refuser une offre dans l'interface contenant les détails de l'offre tels que la Désignation, la Date de création, la Date de début d'offre, la Date de fin d'offre, la Promotion et la Quantité d'offre ?"
response = rag_chain.invoke({"input": query, "chat_history": chat_history})

print(response["answer"])
