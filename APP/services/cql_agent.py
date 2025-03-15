from cassandra_service import CassandraInterface
import os
from TEXT2CQL_PROMPT import TEXT2CQL_PROMPT,ANSWER_PROMPT,PROMPT_FIX_CQL_V2,schema_summary_prompt,EXPLAIN_CQL_ERROR
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google import genai
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
load_dotenv()  # take environment variables from .env.
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY")
from langchain_upstage import ChatUpstage

from langchain_core.messages import HumanMessage
class Query(BaseModel):
    query: str = Field(description="The CQL query that needs to be executed")
    explanation: str = Field(description="A short description of the purpose of the query and how it helps answer the question")
    result_for_next_query: str =Field(description="Indicates what value from the current query will be used in the next query")
class AnswerPrompt(BaseModel):
    queries: list[Query] = Field(description="Contains a list of queries to be executed in sequence")
parser = JsonOutputParser(pydantic_object=AnswerPrompt)
class CQLAGENT:
    def __init__(self):
        self.cassandra_interface = CassandraInterface()

        self.client = genai.Client(api_key="AIzaSyAcIGFo53M8vf2eb_UO4JGBYb0an7B8xH4").chats.create(model="gemini-2.0-flash")
        
        
        self.schema, self.partition_keys, self.clustering_keys = self.generate_schema_partition_clustering_keys(keyspace="shop")
    def generate_schema_partition_clustering_keys(self,keyspace: str = "default_keyspace") -> tuple[str, str, str]:
        """Generates a TEXT2CQL_PROMPT compatible schema for a keyspace"""
        # Get all table names in our keyspace
        
        exist = self.is_schema_up_to_date(keyspace)

        if exist:
            schema_with_summary, partition_keys, clustering_keys = self.fetch_schema_details(keyspace)
            self.save_schema_to_files(keyspace, schema_with_summary, partition_keys, clustering_keys)
            return schema_with_summary, partition_keys, clustering_keys
        else:
            return self.load_existing_schema()

    def fetch_schema_details(self, keyspace):
        table_names =  self.cassandra_interface.execute_statement(
                f"SELECT table_name, comment FROM system_schema.tables WHERE keyspace_name = '{keyspace}'"
            )
        tn_str = ", ".join(["'" + tn.table_name + "'" for tn in table_names])

        columns = self.cassandra_interface.execute_statement(
                f"SELECT * FROM system_schema.columns WHERE table_name IN ({tn_str}) AND keyspace_name = '{keyspace}' ALLOW FILTERING"
            )
            
        schema_details = " | ".join([
                    f"{keyspace}.{table.table_name} '{table.comment}' : " + " , ".join([
                        f"{col.column_name} ({col.type})"
                        for col in columns
                        if col.table_name == table.table_name
                    ])
                    for table in table_names
                ])
        
        schema_summary = self.generate_schema_summary(schema_details)
        schema_with_summary = f"{schema_details}\n\nSummary: {schema_summary}"
        partition_keys = " | ".join([
                f"{keyspace}.{table.table_name} : " + " , ".join([
                    col.column_name for col in columns
                    if col.table_name == table.table_name
                    and col.kind == "partition_key"
                ])
                for table in table_names
            ])
        clustering_keys = " | ".join([
                f"{keyspace}.{table.table_name} : " + " , ".join([
                    f"{col.column_name} ({col.clustering_order})" for col in columns
                    if col.table_name == table.table_name
                    and col.kind == "clustering"
                ])
                for table in table_names
            ])
        
        return schema_with_summary,partition_keys,clustering_keys

    def generate_schema_summary(self, schema_details):
        prompt = PromptTemplate(
                    input_variables=["schema"],
                    template=schema_summary_prompt,
                )
                
        llm_chain = LLMChain(llm=self.agent.llm_gemini, prompt=prompt)
        schema_summary = llm_chain.run({
                    "schema": schema_details,
                })
                
        return schema_summary

    def save_schema_to_files(self, keyspace, schema_with_summary, partition_keys, clustering_keys):
        with open("keyspace.txt", "w") as f:
             f.write(keyspace)
        # Write the schema to a file because it 's expensive to generate

        with open("schema.txt", "w") as f:
             f.write(schema_with_summary
            )
        with open("partition_keys.txt", "w") as f:
            f.write(partition_keys)
        with open("clustering_keys.txt", "w") as f:
            f.write(clustering_keys)

    def is_schema_up_to_date(self, keyspace):
        try:
            with open("keyspace.txt", "r") as f:
                keyspace_prv = f.read().strip()
        except FileNotFoundError:
            keyspace_prv = ""
        exist=keyspace_prv != keyspace
        return exist

    def load_existing_schema(self):
        with open("schema.txt", "r") as f:
            schema_with_summary = f.read()
        with open("partition_keys.txt", "r") as f:
            partition_keys = f.read()
        with open("clustering_keys.txt", "r") as f:
            clustering_keys = f.read()
        return schema_with_summary, partition_keys, clustering_keys
        


    def execute_query_from_question(self,question: str, debug_cql: bool = True, debug_prompt: bool = True, return_cql: bool = False):
        """Generates and executes CQL from a user question based on LLM output"""
        
      
        if debug_prompt:
            print(f"Prompting model with:\n{prompt}")
        prompt = TEXT2CQL_PROMPT.format(
        schema=self.schema,
        partition_keys=self.partition_keys,
        clustering_keys=self.clustering_keys,
        question=question,
         )

        
        completion = self.client.send_message(prompt).text
        completion=completion.replace("```","").replace("cql","").strip()

        if debug_cql:
            print(f"Question: {question}\nGenerated Query: {completion}\n")

        # Need to trim trailing ';' if present to work with cassandra-driver
        if completion.find(";") > -1:
            completion = completion[:completion.find(";")]

        results = self.cassandra_interface.execute_statement(completion)
        if return_cql:
            return (results, completion)
        else:
            return results

    def execute_cql_query(self, question, debug_cql, return_cql, cql):
        completion = cql.replace("```", "").replace("sql", "").replace("\n", "").strip()
        if debug_cql:
            print(f"Question: {question}\nGenerated Query: {completion}\n")

        if completion.find(";") > -1:
            completion = completion[:completion.find(";")]
        completion = completion.replace('`', '')

        results = self.cassandra_interface.execute_statement(completion)
    
        while (isinstance(results,dict)):
            error_message=results.get("error")
            statement=results.get("statement")
            prompt = PROMPT_FIX_CQL_V2.format(
                question=question,
                error_message=error_message,
                statement=statement,
                schema=self.schema,
                partition_keys=self.partition_keys,
                clustering_keys=self.clustering_keys,
            )
            new_reponse=self.agent.chat_up.invoke([HumanMessage(content=prompt)]).content
            new_reponse=new_reponse.replace("```", "").replace("sql", "").replace("\n", "").strip()
            if new_reponse.find(";") > -1:
                new_reponse = new_reponse[:new_reponse.find(";")]

            results = self.cassandra_interface.execute_statement(new_reponse)
            
        if return_cql:
            return results, completion
    def answer_question(self,question: str, debug_cql: bool = True, debug_prompt: bool = False) -> str:
        """Conducts a full RAG pipeline where the LLM retrieves relevant information
        and references it to answer the question in natural language.
        """
        query_results, cql = self.execute_query_from_question(
            question=question,
            debug_cql=debug_cql,
            debug_prompt=debug_prompt,
            return_cql=True,
        )
        prompt = ANSWER_PROMPT.format(
            question=question,
            results_repr=str(query_results),
            cql=cql,
        )

        if debug_prompt:
            print(f"Prompting model with:\n{prompt}")

        messages = [
            HumanMessage(content=prompt),
        ]
        return self.client.invoke(messages).content
