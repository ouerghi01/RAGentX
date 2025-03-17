######## TEXT2CQL_PROMPT ########
TEXT2CQL_PROMPT = """Convert the question to CQL (Cassandra Query Language) that can retrieve an appropriate answer, or answer saying that the data model does not support answering such a question in a performant way:
its No SQL, so you can't use joins, subqueries, or aggregations.
[Schema : values (type)]
{schema}

[Partition Keys]
{partition_keys}

[Clustering Keys]
{clustering_keys}

[Q]
{question}

[CQL]
"""

schema_summary_prompt = """
                    Given the following database schema details:
                    {schema}
                    Provide a concise summary that describes:
                    1. The purpose of each table.
                    2. A description of each column in every table, including:
                    - What the column represents (its meaning).
                    - The data type and how it relates to the table's functionality.
                    - Any key columns or special columns (e.g., primary keys, foreign keys).
                    3. Any relevant relationships between tables (if applicable).
                    Make sure to include a description of each attribute in the table for clarity.
                    """


######## ANSWER_PROMPT ########

ANSWER_PROMPT = """Query:
```
{cql}
```

Output:
```
{results_repr}
```
===

Given the above results from querying the DB, answer the following user question.
When answering the question don't answer in a technical way, hence answer as if you're talking to a layman.

{question}
"""
PROMPT_FIX_CQL_V2 = """
You are an AI agent tasked with fixing a Cassandra Query Language (CQL) query based on:


## **Input Details**

### **Error Message from Database:**  
{error_message}  

### **Schema Details:**  
- **Table Structure:**  
  {schema}  

- **Partition Keys:**  
  {partition_keys}  

- **Clustering Keys:**  
  {clustering_keys}  

### **Original CQL Statement:**  
```cql
{statement}

Generate a corrected CQL query that addresses the error and aligns with the schema structure.
[CQL]

"""
EXPLAIN_CQL_ERROR = """
CQL Error: {error_message}

## Schema Details:
- **Table Structure:**  
  {schema}

- **Partition Keys:**  
  {partition_keys}

- **Clustering Keys:**  
  {clustering_keys}

## Problematic CQL Statement:
{cql_statement}

### Explanation:
The issue in the generated CQL query is identified above. Please analyze the error message and schema details to determine the cause.

### Guidance:
To resolve this issue:
1. Verify that the table structure aligns with the query.
2. Ensure partition and clustering keys are correctly used.
3. Check for syntax errors or missing fields.
4. Review data types and constraints.

Generate a corrected CQL query that addresses the error and aligns with the schema structure.

"""
