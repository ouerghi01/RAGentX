from uuid import uuid4
from cassandra.cluster import Cluster
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import *
# Initialize Cassandra session and set keyspace
session = initialize_database_session()
session.set_keyspace('store_key')

# Create table if it does not exist
create_table_query = """
CREATE TABLE IF NOT EXISTS products (
    product_id UUID PRIMARY KEY,
    product_name TEXT,
    price DECIMAL
);
"""
session.execute(create_table_query)

# Insert sample products into the table
products = [
    ('Laptop', 999.99),
    ('Smartphone', 499.49),
    ('Tablet', 299.99),
    ('Desktop Computer', 1299.99),
    ('Smart Watch', 199.99)
]

for name, price in products:
    session.execute(
        "INSERT INTO products (product_id, product_name, price) VALUES (%s, %s, %s)",
        (uuid4(), name, price)
    )

# Load documents from Cassandra using a custom loader
loader = CassandraLoader(
    table="products",
    session=session,
    keyspace="store_key",
)
docs = loader.load()

# Add source metadata for each document
for doc in docs:
    if doc is not None:
        doc.metadata['source'] = 'cassandra:store_key.products'

# Create retriever from the loaded documents
retriever = create_retriever_from_documents(session, docs)

# Define the language model (using Ollama in this example)
llm = Ollama(model="qwen2.5:3b", base_url="http://localhost:11434")


prompt = """
🔹 **Présentation de l'agent** 🔹  
Je suis un agent intelligent conçu pour fournir des réponses précises et pertinentes en m'appuyant exclusivement sur les informations contenues dans le contexte fourni, qui inclut des documents extraits de la base de données. Mon objectif est d'offrir des explications claires et concises, tout en respectant les limites des données disponibles.  

📌 **Directives pour formuler les réponses :**  
1. Utilise uniquement le contexte et les documents fournis ci-dessous pour formuler ta réponse.  
2. Si l'information demandée n'est pas présente dans le contexte, réponds par "Je ne sais pas".  
3. Fournis des réponses concises, ne dépassant pas trois phrases.  
4. Si le contexte mentionne un outil ou une fonctionnalité spécifique d'un site web ou d'une plateforme SaaS, explique son utilisation et son objectif.  
5. N'ajoute aucune information qui ne soit pas incluse dans le contexte fourni.  
6. Si possible, donne une recommandation pertinente.  

Contexte et Documents: {context}

Question: {question}

Réponse:
"""


QA_CHAIN_PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question"])
document_prompt = PromptTemplate(
        template="Context:\ncontent:{page_content}\nsource:{source}",  
        input_variables=["page_content", "source"]  
    )
llm_chain = LLMChain(
        llm=llm, 
        prompt=QA_CHAIN_PROMPT, 
        callbacks=None, 
        verbose=True
    )
combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,  
        document_variable_name="context",
        callbacks=None,
        document_prompt=document_prompt  
    )
qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,  
        retriever=retriever,
        verbose=True
    )

# Example question to test the chain
question = "Analyze the product prices in the database to determine the average price. Then, list which products are priced above the average and which are priced below it. Explain your analysis process."
response = qa.run(question)

print("Answer:", response)
