
from collections import defaultdict
from cassandra.cluster import Cluster
import cassio
import uuid
from dotenv import load_dotenv
import os
import threading
from google import genai
from datetime import datetime, timedelta
from services.producer import Producer
from services.User import User
load_dotenv()  # take environment variables from .env.
from datetime import datetime
def categorize_timestamp(ts):
    if(ts is None):
        return None
    now = datetime.now()
    today = now.date()
    session_date = ts.date()  # Assuming ts is a datetime object

    if session_date == today:
        return "Today"
    elif session_date == today - timedelta(days=1):
        return "Yesterday"
    elif session_date >= today - timedelta(days=7):
        return "Last 7 Days"
    elif session_date >= today - timedelta(days=30):
        return "Last 30 Days"
    else:
        return session_date.strftime("%B %Y")  # e.g., "February 2025"

class CassandraManager:
    """A class to handle interactions with a Cassandra database.
    This class provides an interface for managing connections, tables, and operations
    with a Cassandra database, specifically designed for storing and retrieving
    document vectors, session information, and chat responses.
    Attributes:
        CASSANDRA_PORT (str): Port number for Cassandra connection
        CASSANDRA_USERNAME (str): Host address for Cassandra connection
        KEYSPACE (str): Keyspace name for the Cassandra database
        session: Cassandra session object
    Methods:
        initialize_database_session(port, host): Initializes connection and creates required tables
        clear_tables(): Clears all data from the tables in the keyspace
        retrieve_column_descriptions(): Gets schema information for all tables
        create_room_session(): Creates a new chat session
        evaluate_response(partition_id, evaluation): Updates evaluation status of a response
        find_same_response(question): Searches for existing responses to a question
        get_chat_history(session_id): Retrieves chat history for a session
        insert_answer(session_id, question, final_answer): Inserts a new Q&A pair
    """
    def __init__(self):
        self.CASSANDRA_PORT=os.getenv("CASSANDRA_PORT")
        self.CASSANDRA_USERNAME=os.getenv("CASSANDRA_HOST")
        self.KEYSPACE:str=os.getenv("KEYSPACE")
        self.session=self.initialize_database_session(self.CASSANDRA_PORT,self.CASSANDRA_USERNAME)
        #self.clean()
        self.create_database_tables()
        self.producer=Producer(
            topic="cassandrastream"
        )
    def get_session(self):
        """
        Returns the current Cassandra session object.

        Returns:
            session: The current Cassandra session object
        """
        return self.session
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method that ensures proper shutdown of the Cassandra session.

        Args:
            exc_type: The type of the exception that was raised, if any
            exc_val: The instance of the exception that was raised, if any
            exc_tb: The traceback of the exception that was raised, if any

        Returns:
            None

        Note:
            This method is automatically called when exiting a context manager block.
            It properly closes the Cassandra session connection.
        """
        print(f"{exc_type} -- {exc_val} -- {exc_tb}")
        self.session.shutdown()  

    def initialize_database_session(self,port,host):
        
        """
        Initialize a Cassandra database session and create necessary tables.
        This method establishes a connection to a Cassandra cluster and initializes the database
        structure by creating a keyspace and required tables if they don't already exist.
        Tables created:
        - vectores: Stores document text, content and vector embeddings
        - session_table: Manages user sessions
        - response_session: Links sessions with responses
        - response_table: Stores Q&A interactions with timestamps and evaluations
        Args:
            port (int): The port number for the Cassandra connection
            host (str): The hostname or IP address of the Cassandra server
        Returns:
            None
        Raises:
            cassandra.cluster.NoHostAvailable: If unable to connect to the Cassandra host

        
        """
        
        self.session = Cluster([host],port=port).connect()
        cassio.init(session=self.session, keyspace=self.KEYSPACE)
            
        create_key_space = f"""
        CREATE KEYSPACE IF NOT EXISTS {self.KEYSPACE}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 3}};
        """
        self.session.execute(create_key_space)
        return self.session

    def create_database_tables(self):
        #remove all tables
        
        create_documents_table = f"""
        CREATE TABLE IF NOT EXISTS {self.KEYSPACE}.documents (
            document_id UUID PRIMARY KEY,
            FILE_name TEXT  -- Text extracted from the PDF
        );
        """
        self.session.execute(create_documents_table)

        create_user_table = f"""
        CREATE TABLE IF NOT EXISTS {self.KEYSPACE}.users_new (
            user_id UUID PRIMARY KEY,
            username TEXT,
            email TEXT,
            his_job TEXT,
            password TEXT
        );
        """
        self.session.execute(create_user_table)
        create_session_table =f"""
            CREATE TABLE IF NOT EXISTS {self.KEYSPACE}.session_table (
                session_id UUID,
                title TEXT,
                timestamp TIMESTAMP,
                close_session_time TIMESTAMP,
                
                user_id UUID,
                PRIMARY KEY (session_id,timestamp)
            )WITH CLUSTERING ORDER BY (timestamp DESC)
            ;
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
            the_time_question_sended TIMESTAMP,
            question TEXT,  
            answer TEXT,
            timestamp TIMESTAMP,
            evaluation BOOLEAN
        );
        """
        #self.session.execute(f"ALTER TABLE {self.KEYSPACE}.response_table WITH cdc = true;")
        self.session.execute(response_table)
        

    def clean(self):
        query=f"SELECT table_name FROM system_schema.tables WHERE keyspace_name='{self.KEYSPACE}'"
        rows=self.session.execute(query)
        for row in rows:
            if row.table_name == "vectores_new":
                continue
            query=f"DROP TABLE {self.KEYSPACE}.{row.table_name}"
            self.session.execute(query)
    def insert_user(self,user_name,user_email,his_job,user_password):
        user_id = uuid.uuid4()
        self.session.execute(
            f"INSERT INTO {self.KEYSPACE}.users_new (user_id, username, email,his_job,password) VALUES (%s, %s, %s, %s, %s)",
            (user_id, user_name, user_email,his_job,user_password)
        )
        return user_id
    def find_user(self,username):
        find_user_query = f"SELECT * FROM {self.KEYSPACE}.users_new WHERE username=%s ALLOW FILTERING"
        result_set = self.session.execute(find_user_query, (username,))
        user_row = result_set.one() if result_set else None
        return user_row
    def execute_statement(self,statement: str):
   
        try:
            rows  = self.session.execute(statement)
            
            return rows
        except Exception as e:
            print(f"Query Failed: {statement} error {e}" )
            return {
                "error": str(e),
                "statement":statement
            }
    def clear_tables(self):
        """
        Clears all tables in the Cassandra database by executing TRUNCATE commands.

        This method truncates the following tables in the specified keyspace:
        - vectores_new
        - response_table
        - response_session
        - session_table

        Note: session_table appears to be truncated twice in the implementation.

        Returns:
            None
        """
        #self.session.execute(f"TRUNCATE {self.KEYSPACE}.vectores_new")
        self.session.execute(f"TRUNCATE {self.KEYSPACE}.response_table")
        self.session.execute(f"TRUNCATE {self.KEYSPACE}.response_session")
        self.session.execute(f"TRUNCATE {self.KEYSPACE}.session_table")
        self.session.execute(f"TRUNCATE {self.KEYSPACE}.session_table")

        #vectores_new

    def retrieve_column_descriptions(self):
        """
        Retrieves column descriptions for all tables in the specified keyspace.

        This method queries the Cassandra system schema to get detailed information about
        columns in all tables within the keyspace defined in self.KEYSPACE.

        Returns:
            dict: A dictionary where:
                - keys are table names (str)
                - values are strings containing column descriptions in the format:
                  "description: column_name: <name> type: <type>"
                  for each column in the table

        Example returned schema format:
            {
                'table1': 'description:\n column_name: id type: uuid\n column_name: name type: text\n',
                'table2': 'description:\n column_name: email type: text\n ...'
            }
        """
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
    def create_room_session(self,username):
        session_id = uuid.uuid4()
        
        user_id = self.retrieve_user_id(username)
        if user_id is None:
            raise ValueError("User not found")
        timestamp = datetime.now()
        create_session_row=f"""INSERT INTO {self.KEYSPACE}.session_table (session_id,user_id,timestamp) VALUES (%s,%s,%s)"""
        self.session.execute(create_session_row,(session_id,user_id,timestamp))
        return session_id
    def evaluate_reponse(self,partition_id,evaluation):
        query=f"UPDATE {self.KEYSPACE}.response_table SET evaluation={evaluation} WHERE partition_id={partition_id}"
        self.session.execute(query)
    def find_same_reponse(self,question):
        find_reponse_query=f"SELECT * FROM {self.KEYSPACE}.response_table WHERE question=%s  ALLOW FILTERING"
        result_set = self.session.execute(find_reponse_query, (question,))
        response_row = result_set.one() if result_set else None
        return response_row
    def get_sessions(self,user):
        user_id=self.retrieve_user_id(user)
        all_sessions = self.session.execute(f"SELECT session_id, title, timestamp FROM {self.KEYSPACE}.session_table   WHERE user_id=%s ALLOW FILTERING", (user_id,))
        if not all_sessions:
            return {}
        all_sessions = {row.session_id: {"title":row.title,"timestamp":row.timestamp} for row in all_sessions}
        grouped_sessions = defaultdict(list)
        for key,row in all_sessions.items():  # Assuming all_sessions is a list of session objects
            category = categorize_timestamp(row['timestamp'])
            if category is None:
                continue
            if row['title'] is not None:
                grouped_sessions[category].append({
                    "session_id": key,
                    "title": row['title'],
                    "timestamp": row['timestamp']
                })
        sort_order = ["Today", "Yesterday", "Last 7 Days", "Last 30 Days"]
        sorted_grouped = {}
        for key in sort_order:
            if key in grouped_sessions:
                sorted_grouped[key] = grouped_sessions[key]
        for key in grouped_sessions:
            if key not in sort_order:
                sorted_grouped[key] = grouped_sessions[key]
        return sorted_grouped
    def close_room_session(self,session_id):
        session = self.get_last_session_created(session_id)
        if session is None:
            raise ValueError("Session not found")
        close_session_time=datetime.now()
        self.session.execute(
            f"Insert into {self.KEYSPACE}.session_table (session_id,title,timestamp,user_id,close_session_time) VALUES (%s,%s,%s,%s,%s)",
            (session_id, session.title, session.timestamp, session.user_id,close_session_time)
        )

    def get_last_session_created(self, session_id):
        query = f"""
            SELECT * FROM {self.KEYSPACE}.session_table
            WHERE session_id=%s LIMIT 1
        """
        session = self.session.execute(query, (session_id,)).one()
        return session
    def get_last_session_created_by_user(self, username):
        user_id = self.retrieve_user_id(username)
        query = f"""
            SELECT * FROM {self.KEYSPACE}.session_table
            WHERE user_id=%s LIMIT 1 ALLOW FILTERING
        """
        session = self.session.execute(query, (user_id,)).one()
        return session
    def get_chat_history(self, session_id):
        session_id= uuid.UUID(session_id)
        query = f"""
            SELECT * FROM {self.KEYSPACE}.session_table
            WHERE session_id=%s LIMIT 1
        """
        session = self.session.execute(query, (session_id,)).one()
        timestamp=datetime.now()
        self.session.execute(
            f"Insert into {self.KEYSPACE}.session_table (session_id,title,timestamp,user_id) VALUES (%s,%s,%s,%s)",
            (session_id, session.title, timestamp, session.user_id)
        )
        
        all_ids = self.session.execute(f"SELECT partition_id FROM {self.KEYSPACE}.response_session WHERE session_id=%s ALLOW FILTERING", (session_id,))
        all_ids = [row.partition_id for row in all_ids]
        result_set = []
        for id in all_ids:
            query_sql = f"SELECT question, answer FROM {self.KEYSPACE}.response_table WHERE partition_id = %s LIMIT 5 ALLOW FILTERING"
            result_set.extend(self.session.execute(query_sql, (id,)))
        chat_history = []
        for row in result_set:
            chat_history.append((row.question, row.answer))
        return chat_history
    def insert_answer(self,session_id,user:User,question,final_answer,time_sended):
        
        partition_id = uuid.uuid1()
        now=datetime.now()
        self.insert_and_produce(question, final_answer, partition_id, now,time_sended,user.username)
        get_session=f"SELECT * FROM {self.KEYSPACE}.session_table WHERE session_id=%s ALLOW FILTERING"
        result_set = self.session.execute(get_session, (session_id,))
        session_row = result_set.one() if result_set else None
        if session_row.title is None :
            llm=genai.Client(api_key="AIzaSyAcIGFo53M8vf2eb_UO4JGBYb0an7B8xH4").chats.create(model="gemini-2.0-flash")
            prompt = f"Generate a short and engaging chat title based on this question: '{question}'. Respond with a maximum of 3 lowercase words, separated by commas, without any extra symbols or formatting."
            title=llm.send_message(prompt).text
            
            query=f"UPDATE {self.KEYSPACE}.session_table SET title=%s WHERE session_id=%s and timestamp=%s"
            now = datetime.now()
            self.session.execute(query,(title, session_id,session_row.timestamp)) 

        query_session_response_related=f"""INSERT INTO {self.KEYSPACE}.response_session (session_id,partition_id) VALUES (%s,%s)"""
        self.session.execute(query_session_response_related,(session_id,partition_id))

    def insert_and_produce(self, question, final_answer, partition_id, now,time_sended,user):
        thread = threading.Thread(target=self.insert_into_reponse_table, args=(question, final_answer, partition_id, now,time_sended))
        thread.start()

        producer_thread = threading.Thread(
            target=self.producer.poll_cassandra,
            kwargs={
                "timestamp": now,
                "partition_id": partition_id,
                "answer": final_answer,
                "question": question,
                "the_time_question_sended": time_sended,
                "user": user
            }
        )
        producer_thread.start()

        thread.join()
        producer_thread.join()

    def insert_into_reponse_table(self, question, final_answer, partition_id, now,time_sended):
        self.session.execute(
        f"INSERT INTO {self.KEYSPACE}.response_table (partition_id, question, answer,timestamp,evaluation,the_time_question_sended) VALUES (%s, %s, %s, %s,false,%s)",
        (partition_id,question, final_answer,now,time_sended)
        )

    def retrieve_user_id(self, user: User | str):
        username = user if isinstance(user, str) else user.username
        user_result = self.session.execute(
            f"SELECT user_id FROM {self.KEYSPACE}.users_new WHERE username=%s ALLOW FILTERING", 
            (username,)
        ).one()
        user_id = user_result.user_id if user_result else None
        return user_id
    
     