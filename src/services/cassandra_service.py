
from cassandra.cluster import Cluster
import cassio
import uuid
from dotenv import load_dotenv
import os 
load_dotenv()  # take environment variables from .env.
from datetime import datetime
class CassandraInterface:
    def __init__(self):
        self.CASSANDRA_PORT=os.getenv("CASSANDRA_PORT")
        self.CASSANDRA_USERNAME=os.getenv("CASSANDRA_HOST")
        self.KEYSPACE=os.getenv("KEYSPACE")
        self.initialize_database_session(self.CASSANDRA_PORT,self.CASSANDRA_USERNAME)

        

    def initialize_database_session(self,port,host):
        
        self.session = Cluster([host],port=port).connect()
        cassio.init(session=self.session, keyspace=self.KEYSPACE)
            
        create_key_space = f"""
        CREATE KEYSPACE IF NOT EXISTS {self.KEYSPACE}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 3}};
        """
        self.session.execute(create_key_space)
        create_table = f"""
        CREATE TABLE IF NOT EXISTS {self.KEYSPACE}.vectores (
            partition_id UUID PRIMARY KEY,
            document_text TEXT,  -- Text extracted from the PDF
            document_content BLOB,  -- PDF content stored as binary data
            vector BLOB  -- Store the embeddings (vector representation of the document)
        );
        """
        create_session_table = f"""

        CREATE TABLE IF NOT EXISTS {self.KEYSPACE}.session_table (
            session_id UUID PRIMARY KEY,
        );
        """
        self.session.execute(create_session_table)
        session_response_table = f"""
        CREATE TABLE IF NOT EXISTS {self.KEYSPACE}.response_session (
            partition_id UUID PRIMARY KEY,
            session_id UUID,
            table_response_id UUID,
        );
        """
        self.session.execute(session_response_table)
        
        response_table = f"""
        CREATE TABLE IF NOT EXISTS {self.KEYSPACE}.response_table (
            partition_id UUID PRIMARY KEY,
            question TEXT,  
            answer TEXT,
            timestamp TIMESTAMP,
            evaluation BOOLEAN
            
        );
        """
        self.session.execute(create_table)
        self.session.execute(response_table)
    def clear_tables(self):
        self.session.execute(f"TRUNCATE {self.KEYSPACE}.vectores")
        self.session.execute(f"TRUNCATE {self.KEYSPACE}.response_table")
        self.session.execute(f"TRUNCATE {self.KEYSPACE}.response_session")
        self.session.execute(f"TRUNCATE {self.KEYSPACE}.session_table")

    def retrieve_column_descriptions(self):
        query_schema = f"""
        SELECT * FROM system_schema.columns 
        WHERE keyspace_name = '{self.KEYSPACE}';
         """
        rows=self.session.execute(query_schema)
        schema={}
        for row in rows:
            if schema.get(row.table_name) is None:
                schema[row.table_name]=f"description : \n column_name : {row.column_name}  type : {row.type} \n"
            else:
                schema[row.table_name]+=f" column_name : {row.column_name}  type : {row.type} \n"
        return schema
    def create_room_session(self):
        session_id = uuid.uuid4()
        create_session_row=f"""INSERT INTO {self.KEYSPACE}.session_table (session_id) VALUES (%s)"""
        self.session.execute(create_session_row,(session_id,))
        print("session created")
        return session_id
    def evaluate_reponse(self,partition_id,evaluation):
        query=f"UPDATE {self.KEYSPACE}.response_table SET evaluation={evaluation} WHERE partition_id={partition_id}"
        self.session.execute(query)
    def find_same_reponse(self,question):
        find_reponse_query=f"SELECT * FROM {self.KEYSPACE}.response_table WHERE question=%s  ALLOW FILTERING"
        result_set = self.session.execute(find_reponse_query, (question,))
        response_row = result_set.one() if result_set else None
        return response_row
    def get_chat_history(self, session_id):
        all_ids = self.session.execute(f"SELECT partition_id FROM {self.KEYSPACE}.response_session WHERE session_id=%s ALLOW FILTERING", (session_id,))
        all_ids = [row.partition_id for row in all_ids]
        result_set = []
        for id in all_ids:
            query_sql = f"SELECT question, answer FROM {self.KEYSPACE}.response_table WHERE partition_id = %s LIMIT 5 ALLOW FILTERING"
            result_set.extend(self.session.execute(query_sql, (id,)))
        chat_history = []
        for row in result_set:
            chat_history.append(("human", row.question))
            chat_history.append(("assistant", row.answer))
        if len(chat_history) == 0:
            chat_history.append(("human", ""))
            chat_history.append(("assistant", ""))
        return chat_history
    def insert_answer(self,session_id,question,final_answer):
        partition_id = uuid.uuid1()
        now=datetime.now()
        self.session.execute(
        f"INSERT INTO {self.KEYSPACE}.response_table (partition_id, question, answer,timestamp,evaluation) VALUES (%s, %s, %s, %s,false)",
        (partition_id,question, final_answer["answer"],now)
        )
        query_session_response_related=f"""INSERT INTO {self.KEYSPACE}.response_session (session_id,partition_id) VALUES (%s,%s)"""
        self.session.execute(query_session_response_related,(session_id,partition_id))
        print("answer inserted")
    
     