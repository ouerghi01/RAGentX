
import os
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
import pandas as pd 
import json 
def load_pdf_documents(bool=True):
        if bool==True:
            docs=[]
            dir_path = Path("C:/Users/Ons/IA-Ollama-RAG-IMPL/src/uploads")
            for file in os.listdir(dir_path):
                if file.endswith(".pdf"):
                    full_path = dir_path/ file
                    loader = PDFPlumberLoader(full_path)
                    document = loader.load()
                    docs.extend(document)
            return docs
        else:
            return []
            
            
        #self.load_documents_from_cassandra(documents)

def collect_data(load_pdf_documents) -> None:
    DOCS:List[Document]=load_pdf_documents(True)
    prompt = """
    You are provided with a block of text content. 
    Your task is to extract key details from the content and generate five pairs of questions and their corresponding responses based solely on that text. 
    Structure your answer as a list of objects where each object has two fields: one for the question and one for the response. 
    Do not include any additional commentary or explanation. Each question must be specific to the content, and each response should accurately answer its question. 
    Do not mention the name of the data format or reveal any formatting details—simply provide the structured output.
    {page_content}
    """

    llm=Ollama(model="llama3:8b", base_url="http://localhost:11434",verbose=True,callback_manager=None)
    QA_CHAIN_PROMPT = PromptTemplate(template=prompt, input_variables=["page_content"])
    llm_chain = LLMChain(
            llm=llm, 
            prompt=QA_CHAIN_PROMPT, 
            callbacks=None, 
            verbose=True
        )
    df = pd.DataFrame()
    for i, doc in enumerate(DOCS):
         content=doc.page_content
         retrieval_query = llm_chain.invoke({"page_content": content})
         try:
            structured_output = json.loads(retrieval_query['text'])
         except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing document {i}: {e}")
            df.to_csv("structured_output.csv", index=False)
            return
         new_df=pd.DataFrame(structured_output)
         df = pd.concat([df, new_df], axis=0) 
    columns=['question','response']
    clean_data=df[columns].dropna()
    clean_data.to_csv("Distillation_Data.csv", index=False)
   
# add Explanation to every question and response by llm  
def get_explanation(question, answer):
    llm=Ollama(model="qwen2.5:0.5b", base_url="http://localhost:11434",verbose=True,callback_manager=None)

    prompt = """
    You are provided with a 2 block of text content. one for question and one for answer 
    Your task is to provide an explanation for each question and answer pair.
    {question}
    {answer}
    you need to provide explanation:
    """

    QA_CHAIN_PROMPT = PromptTemplate(template=prompt, input_variables=["question","answer"])
    llm_chain = LLMChain(
            llm=llm, 
            prompt=QA_CHAIN_PROMPT, 
            callbacks=None, 
            verbose=True
)
    output = llm_chain.invoke({"question": question, "answer": answer})['text']
    return output

#https://mail.google.com/mail/u/0/#inbox/FMfcgzQZTMMzrpJKZvtMbzMLcWmwRlrR
#https://medium.com/@jyotidabass/deep-learning-simplified-a-beginners-guide-to-knowledge-distillation-with-python-ce1f4e4f9598
#https://github.com/minyang-chen/Knowledge_Distillation_Training/blob/main/knowledge_distillation_training.ipynb