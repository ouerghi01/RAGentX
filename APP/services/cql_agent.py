from cassandra_service import CassandraInterface
from TEXT2CQL_PROMPT import PROMPT_FIX_CQL_V2
from dotenv import load_dotenv
import json 
from google import genai
load_dotenv()  # take environment variables from .env.
from langchain_core.messages import HumanMessage
class CQLAGENT:
    def __init__(self):
        self.cassandra_interface = CassandraInterface()
        self.client = genai.Client(api_key="AIzaSyAcIGFo53M8vf2eb_UO4JGBYb0an7B8xH4").chats.create(model="gemini-2.0-flash")
        self.schema, self.partition_keys, self.clustering_keys = self.generate_schema_partition_clustering_keys(keyspace="shop")
    def generate_schema_partition_clustering_keys(self,keyspace: str = "default_keyspace") -> tuple[str, str, str]:
        """Generates a TEXT2CQL_PROMPT compatible schema for a keyspace"""
        schema_with_summary, partition_keys, clustering_keys = self.fetch_schema_details(keyspace)
        return schema_with_summary, partition_keys, clustering_keys

    def fetch_schema_details(self, keyspace):
        table_names =  self.cassandra_interface.execute_statement(
                f"SELECT table_name, comment FROM system_schema.tables WHERE keyspace_name = '{keyspace}'"
            )
        table_names = list(table_names)
        tn_str = ", ".join(["'" + tn.table_name + "'" for tn in table_names])

        columns = self.cassandra_interface.execute_statement(
                f"SELECT * FROM system_schema.columns WHERE table_name IN ({tn_str}) AND keyspace_name = '{keyspace}' ALLOW FILTERING"
            )
        columns = list(columns)
        schema_details = " | ".join([
                    f"{keyspace}.{table.table_name} '{table.comment}' : " + " , ".join([
                        f"{col.column_name} ({col.type})"
                        for col in columns
                        if col.table_name == table.table_name
                    ])
                    for table in table_names
                ])
        schema_with_summary = f"{schema_details}\n"
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
    def execute_cql_query_fallback(self, query, resultas):
            error=resultas["error"]
            query_incorrect=query
            schema=self.schema
            partition_keys=self.partition_keys
            clustering_keys=self.clustering_keys
            prompt=PROMPT_FIX_CQL_V2.format(
                    error_message=error,
                    schema=schema,
                    partition_keys=partition_keys,
                    clustering_keys=clustering_keys,
                    statement=query_incorrect
                )
            client=genai.Client(api_key="AIzaSyAcIGFo53M8vf2eb_UO4JGBYb0an7B8xH4").chats.create(model="gemini-2.0-flash")
            completion = client.send_message(prompt).text
            completion=completion.replace("```","").replace("cql","").strip()
            query=completion
            resultas=self.cassandra_interface.execute_statement(completion)
            return resultas
    def answer_data_with_cql(self, question):
        prompt_text="""
            Convert the given question into multiple CQL (Cassandra Query Language) queries that, when executed sequentially, can retrieve an appropriate answer.
            1. **Avoid `GROUP BY`, `JOINs`, and `Subqueries`:**
            - Cassandra does not support these features, so do not use them in any query. 
            - Do not attempt to simulate `JOIN` operations with subqueries or similar techniques.
            2- Do no use Complex queries, keep it simple and direct ,and solve the question in subqueries.
            3- Do not use ORDER BY  in the queries.
            If the question cannot be answered efficiently due to Cassandra's limitations, such as the lack of joins, subqueries, or aggregations, return a response stating that the data model does not support answering this question in a performant way.
            **Database Schema Information:**
            - **Tables and Columns:** schema details will be provided: {schema}
            - **Partition Keys:** partition key details will be provided: {partition_keys}
            - **Clustering Keys:** clustering key details will be provided: {clustering_keys}
            **User Question:**
            A specific user question will be provided:
            {question}
            ---
            ### **Important Notes on Query Generation:**
            2. **Simple and Direct Queries:**
            - Generate simple queries that focus on retrieving a single piece of data at a time. 
            - Avoid making queries overly complex or long.
            3. **Retrieve Necessary IDs First:**
            - Ensure that you first retrieve necessary identifiers like `user_id` or `product_id` using a query before filtering on them in subsequent queries.
            4. **Use `ALLOW FILTERING` Only When Required:**
            - If the query requires filtering on non-primary key columns, use `ALLOW FILTERING`, but only if necessary for the query to execute.
            ---
            ### **Output Format:**
            Return a JSON object containing a list of queries under the key `"queries"`. Each query object should include:
            - `"query"`: The CQL query to be executed. Make sure the query is simple, direct, and efficient.
            - `"explanation"`: A brief description of what the query does.
            - `"need_agent"`: A boolean value indicating whether this query requires an external process or agent to use the result of the previous query and generate follow-up queries dynamically.
            **How the `need_agent` Field Works:**
            - Set `"need_agent": true` if the query requires a result from a previous query to dynamically generate or fill in values.
            - Set `"need_agent": false` if the query can be executed independently without needing any additional data.
            ---
            If the question cannot be answered efficiently due to Cassandra’s limitations, return a JSON object with an error message explaining the limitations.
            """

        prompt = prompt_text.format(
        schema=self.schema,
        partition_keys=self.partition_keys,
        clustering_keys=self.clustering_keys,
        question=question,
        )
        messages = [
                HumanMessage(content=prompt),
            ]
        client=genai.Client(api_key="AIzaSyAcIGFo53M8vf2eb_UO4JGBYb0an7B8xH4").chats.create(model="gemini-2.0-flash")
        completion = client.send_message(prompt).text
        completion=completion.replace("```","").replace("json","").strip()
        completion_json=json.loads(completion)
        prev ={
        }
        prev_query=""
        for_next_agent=False
        for _,query_data in enumerate(completion_json["queries"]):
            query=query_data["query"]
            query=query.replace(";","")
            if "WHERE"  in query and "ALLOW FILTERING" not in query:
                query+= "ALLOW FILTERING"
            if for_next_agent:
                prompt="""
            Given the result: {result} from the previous query,  
            The following CQL query:  
            {cql}  
            is missing some required data to execute without errors.  
            Please use the provided result to generate a new CQL query that correctly retrieves the necessary data.  
            Return only the corrected CQL query.  
            [cql]
            """
                result="\n".join([str(i) for i in prev[prev_query]])
                pp=prompt.format(result=result,cql=query)
                client=genai.Client(api_key="AIzaSyAcIGFo53M8vf2eb_UO4JGBYb0an7B8xH4").chats.create(model="gemini-2.0-flash")

                completion = client.send_message(pp).text
                completion=completion.replace("```","").replace("cql","").strip()
                if completion.find(";") > -1:
                    completion = completion[:completion.find(";")]
                resultas=self.cassandra_interface.execute_statement(completion)
                while isinstance(resultas,dict):
                    resultas = self.execute_cql_query_fallback( query, resultas)
                prev[query]=resultas.all()
            
            else:
                resultas=self.cassandra_interface.execute_statement(query)
                while isinstance(resultas,dict):
                    resultas = self.execute_cql_query_fallback( query, resultas)
                prev[query]=resultas.all()
                prev_query=query
            for_next_agent=query_data["need_agent"]
        data_collected=""
        for key in prev:
            data_collected+=f"{key}:\n{prev[key]}\n"
        prompt = """
        Given the following data: 
        {data}
        And the user question: 
        {question}
        Please answer the question based on the provided data.
        [Answer]
        """

        prompt=prompt.format(data=data_collected,question=question)
        answer = self.client.send_message(prompt).text
        return answer
#https://github.com/Hungreeee/Q-RAG/blob/main/demo/demo_notebook_squad.ipynb