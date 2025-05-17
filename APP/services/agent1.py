import asyncio
import os
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from agent_service import AgentInterface
from langchain_core.tools import tool
from sql_agent_service import Sql_agent
from cql_agent import CQLAGENT
llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("LANGSMITH_API_KEY"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
@tool
def choose_next_agent(message: str) -> str:
    """Choose the next agent based on the message content."""
    agents = [
        "CQL agent - Cassandra-based Q&A",
        "SQL agent - SQL-based Q&A",
        "Deep Answering Agent - Retrieval-Augmented Generation",
    ]
    prompt = f"""
You are a system router. Pick the most suitable agent.
Message: "{message}"
Available agents:
- {agents[0]}
- {agents[1]}
- {agents[2]}

Just return one of: "CQL agent", "SQL agent", "Deep Answering Agent".
"""
    response = llm.invoke(prompt)
    return response.content.strip()

agent_service = AgentInterface()
agent_service.compression_retriever=asyncio.run(agent_service.setup_ensemble_retrievers())
agent_service.chain=agent_service.retrieval_chain()
alice = create_react_agent(
    llm,
    tools=[
        choose_next_agent,
        create_handoff_tool(agent_name="SQL agent", description="If the question is SQL-related, hand off to SQL agent."),
        create_handoff_tool(agent_name="CQL agent", description="If the question is CQL-related, hand off to CQL agent."),
        create_handoff_tool(agent_name="Deep Answering Agent", description="If the question is not answered, hand off to RAG agent."),
    ],
    prompt="You are Alice, a router that chooses or hands off to the best agent.",
    name="Alice",
)

sql_llm_tool = Sql_agent(llm)

# Ensure sql_llm_tool is a callable or properly decorated tool
if hasattr(sql_llm_tool, "run"):
    sql_llm_tool_func = sql_llm_tool.run
else:
    sql_llm_tool_func = sql_llm_tool

sql_agent = create_react_agent(
    llm,
    tools=[
        sql_llm_tool_func,
        create_handoff_tool(agent_name="Html agent", description="If the question is answered, send to Html agent to render."),
    ],
    prompt="You are SQL agent, answering using SQL knowledge base.",
    name="SQL agent",
)
cql_tool = CQLAGENT().answer
cql_agent = create_react_agent(
    llm,
    tools=[
        cql_tool,
        create_handoff_tool(agent_name="Deep Answering Agent", description="If not found in database, hand off to RAG agent."),
        create_handoff_tool(agent_name="Html agent", description="If the question is answered, send to Html agent to render."),
    ],
    prompt="You are CQL agent, answering using Cassandra.",
    name="CQL agent",
)
deep_agent = create_react_agent(
    llm,
    tools=[
        agent_service.chain.invoke,
        create_handoff_tool(agent_name="Html agent", description="If the question is answered, send to Html agent to render."),
    ],
    prompt="You are Deep Answering Agent, a RAG-based expert.",
    name="Deep Answering Agent",
)
html_agent = create_react_agent(
    llm,
    tools=[agent_service.evaluate],
    prompt="You are Html agent, format the final answer in HTML.",
    name="Html agent",
)
checkpointer = InMemorySaver()

swarm = create_swarm(
    agents=[alice, sql_agent, cql_agent, deep_agent, html_agent],
    default_active_agent="Alice",
)

app = swarm.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "my-thread-id"}}

response = app.invoke(
    {"messages": [{"role": "user", "content": "What is power Bi ? "}]},
    config,
)

# Extract just the final text
final_answer = response["messages"][-1].content
print("Final Answer:", final_answer)
