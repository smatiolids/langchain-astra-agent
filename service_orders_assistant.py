import os
import json
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish, agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools import Tool
from typing import List
from callbacks import AgentCallbackHandler
from astrapy.db import AstraDB, AstraDBCollection
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import AstraDB as AstraDBVectorStore
from langchain.embeddings import OpenAIEmbeddings
import re

USE_CASE = 'COMPLAINT'
CUSTOMER_ID = 'f08a6894-1863-491d-8116-3945fb915597'  # Mocked for testing
ASTRA_URL = f'{os.environ["ASTRA_API_ENDPOINT"]}/api/rest/v2/keyspaces/{os.environ["ASTRA_KEYSPACE"]}'
astra_db = AstraDB(
    api_endpoint=os.environ["ASTRA_API_ENDPOINT"],
    token=os.environ["ASTRA_TOKEN"],
)

service_orders = AstraDBCollection(
    "service_orders",  astra_db=astra_db)


# TOOL Definition =  Service Orders

class GetServiceOrdersInput(BaseModel):
    _id: str = Field(
        description="The service order number")
    customer_id: str = Field(
        description="The UUID that represents the customer")
    scheduled_date: dict = Field(
        description="Scheduled date for the service")
    execution_date: dict = Field(
        description="Execution date for the service")
    status: dict = Field(
        description="Status of the service order: Not Executed, Executed")


@tool(args_schema=GetServiceOrdersInput)
def get_service_orders(args) -> [str]:
    """Returns information about multiple service orders."""
    filter = args
    print(f"Service Orders condition: {filter}")
    data = service_orders.find(filter=filter, projection={
        "customer_id": 1, "scheduled_date": 1, "execution_date": 1})
    return data['data']['documents']


class GetServiceOrderInput(BaseModel):
    _id: str = Field(
        description="The service order number")

# TOOL Definition =  Scheduled Flight  Detail


@tool(args_schema=GetServiceOrderInput)
def get_service_order_detail(args) -> [str]:
    """Returns information about one service order"""
    print(f"Service detail: {args} {type(args)}")

    filter = {}
    if "_id" in args:
        filter["_id"] = args["_id"]
    elif "order_id" in args:
        filter["_id"] = args["order_id"]
    elif "service_order_id" in args:
        filter["_id"] = args["service_order_id"]

    data = service_orders.find_one(filter=filter)
    return data['data']['document']

# Auxiliary functions


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


def remove_json_comments(json_with_comments):
    """Sometimes, the JSON returned by the LLM contains comments, then it is needed to remove it"""
    comment_pattern = r'/\*.*?\*/|//.*?$'
    json_without_comments = re.sub(
        comment_pattern, '', json_with_comments, flags=re.MULTILINE)
    return json_without_comments


agent = None


tools = [get_service_orders, get_service_order_detail]

astraVectorScore = AstraDBVectorStore(
    embedding=OpenAIEmbeddings(),
    collection_name=f"vector_context",
    token=os.environ["ASTRA_TOKEN"],
    api_endpoint=os.environ["ASTRA_API_ENDPOINT"],
)

retriever = astraVectorScore.as_retriever(
    search_kwargs={"k": 5}
)

retriever_tool = create_retriever_tool(
    retriever,
    "search_qa",
    "Knowledge base for general questions.",
)

tools.append(retriever_tool)

# https://smith.langchain.com/hub/hwchase17/react
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Customer ID: {customer_id}
Language: Portuguese

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action in JSON format without comments.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Write an email to explain to the customer your findings. Sign the email as 'Customer X'

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(tools),
    tool_names=', '.join([t.name for t in tools]),
    customer_id=CUSTOMER_ID
)

llm = ChatOpenAI(temperature=0,
                 model_name='gpt-4-1106-preview',
                 stop=["\nObservation"],
                 callbacks=[AgentCallbackHandler()])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
    }
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)
print("Customer Service Assistant - Initialized")


def generate_response(question):
    agent_step = ""
    intermediate_steps = []
    while not isinstance(agent_step, AgentFinish):
        agent_step: [AgentFinish, AgentAction] = agent.invoke(
            {"input": question,
                "agent_scratchpad": intermediate_steps})

        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            print("Tool input: ", tool_input)
            observation = tool_to_use.func(args={
                **json.loads(remove_json_comments(tool_input))
            }
            )
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(f"Finish: {agent_step.return_values}")

    return agent_step.return_values['output']
