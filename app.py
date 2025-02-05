from fastapi import FastAPI, Depends, HTTPException, Body
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from typing import Annotated, TypedDict, List  # Import for State definition
from langgraph.graph.message import AnyMessage, add_messages  # Import for State definition
import uuid
import os
from dotenv import load_dotenv
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool  # Import the @tool decorator
import requests
from requests.auth import HTTPBasicAuth
import json
from bs4 import BeautifulSoup
import re
from langchain_core.output_parsers import StrOutputParser
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# In-memory storage for user conversations
user_conversations = {}

# Endpoint URL
url = os.environ.get("SCL_URL")
username = os.environ.get("SCL_USERNAME")
password = os.environ.get("SCL_PASSWORD")
shipping_url = os.environ.get("SHIPMENT_TRACKING_URL")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "rag-pinecone-labse"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)

# Load the LaBSE model
embedding_model = SentenceTransformer('sentence-transformers/LaBSE')

# Define the State class
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# Enhanced tool definitions
@tool
def order_information(order_id: str) -> str:
    """Retrieve order and shipping details by order ID and postal code."""
    print("order_information")
    # JSON body
    payload = {
        "action": "getOrderInformation",
        "order_id": order_id
    }

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Send POST request
    response = requests.post(
        url,
        headers=headers,
        auth=HTTPBasicAuth(username, password),
        data=json.dumps(payload)
    )

    # Print response
    print(response.status_code)
    print(response.json())



    return f"Order {order_id} , details: {response.json()}  , shippment_tracking_url: {shipping_url}"

@tool
def voucher_information() -> str:
    """Retrieve voucher related information."""
    print("voucher_information")
    # JSON body
    payload = {
        "action": "getCurrentShopifyVoucherCodes",
    }

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Send POST request
    response = requests.post(
        url,
        headers=headers,
        auth=HTTPBasicAuth(username, password),
        data=json.dumps(payload)
    )

    # Print response
    print(response.status_code)
    print(response.json())



    return f" Voucher Information : {response.json()} "


@tool
def product_information() -> str:
    """Retrieve pricing information for the product."""
    print("product_information")
    # JSON body
    payload = {
        "action": "getCurrentShopifyPrices",
    }

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Send POST request
    response = requests.post(
        url,
        headers=headers,
        auth=HTTPBasicAuth(username, password),
        data=json.dumps(payload)
    )

    # Print response
    print(response.status_code)
    print(response.json())



    return f" Voucher Information : {response.json()} "

@tool
def escalate_to_human(name: str, email: str) -> str:
    """Escalate conversation to human agent. Requires both name and email."""
    print("escalate_to_human", name, email)
    if(name == "" or email == ""):
        return "Please provide both your name and email to escalate the ticket."
    return f"Escalated ticket created for {name} ({email})"

@tool
def knowledgebase_sanaexpert(qq: str) -> str:
    """SanaExpert Product Information, return, shippment policies and general information etc."""
    print("knowledgebase_sanaexpert")
    query_embedding = embedding_model.encode([qq])[0].tolist()
    print(type(pinecone_index))
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    return "\n\n".join([match.metadata["text"] for match in results.matches])

def handle_tool_error(state) -> dict:
    print("handle_tool_error" , state.get("error"))
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

# Define the Assistant class
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            order_id = configuration.get("order_id", None)
            name = configuration.get("name", None)
            email = configuration.get("email", None)
            state = {**state, "user_info": order_id}
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Question rewriter
websystem = """You are a question re-writer that converts an input question to a better version optimized for web search."""
re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", websystem),
    ("human", "Here is the initial question:\n\n{question}\nFormulate an improved question."),
])
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=1)
question_rewriter = re_write_prompt | llm | StrOutputParser()

# Web search tool
web_search_tool = TavilySearchResults(k=3, include_domains=["https://sanaexpert.es/"])

@tool
def web_search(query: str) -> str:
    """
    Perform a web search based on the given query.

    Args:
        query (str): The query for the web search.

    Returns:
        str: A string containing the search results.
    """
    print("web search")
    rewritten_query = question_rewriter.invoke({"question": query}) or ""
    
    # Perform web search
    docs = web_search_tool.invoke({"query": rewritten_query}) or []

    return docs


# Tools list
part_1_tools = [order_information, product_information, knowledgebase_sanaexpert, web_search, escalate_to_human, voucher_information]

# Primary assistant prompt
# Define the primary assistant prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", """ACT LIKE a friendly SanaExpert customer support agent named Maria and respond on behalf of SanaExpert which is a company who deals in food supplements related to maternity, sports, weight control etc. Please Follow these guidelines:

1. Start with a warm greeting and offer help
2. Handle casual conversation naturally without tools
3. For previous order/shipping queries:
   - First ask for order ID (required) if not provided before.
   - Ask for postal code (required)
   - Only provide information about that specific order.
   - After 3 failed attempts, or If the query is about returning or refund specific product collect name and email and escalate to human agent.
4. If the question is about SanaExpert or its products, policies etc get information using SanaExpertKnowledgebase.
5. For up-to-date product prices and product_url use product_information tool. Remember all prices are in euro and for product restock queries, answer that the product will be back in approx 2 weeks.
6. For voucher related queries use voucher_information tool.
7. Use tools ONLY when specific data is needed.
8. Maintain professional yet approachable tone.
9. Clarify ambiguous requests before acting.
10. Keep your response very brief and concise and ask one thing at a time.
11. Use tools information only in the background and don't tell it to the customer. If you can't find any info from knowledgebase and other sources you can try web_search with query.
12. In Case you are not sure about answer just ask customer for his name and email if not provided before. and then tell the user that you are escalating the ticket to human representation and then call escalate_to_human tool."""),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

# Build assistant runnable
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

# Build graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)

# Chat endpoint
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for each user")
    message: str = Field(..., description="User message")

@app.post("/chat")
async def chat(request_data: ChatRequest):
    user_id = request_data.user_id
    user_message = request_data.message

    if not user_id or not user_message:
        raise HTTPException(status_code=400, detail="Both user_id and message are required")

    if user_id not in user_conversations:
        user_conversations[user_id] = {
            "thread_id": str(uuid.uuid4()),
            "history": []
        }

    thread_id = user_conversations[user_id]["thread_id"]
    config = {
        "configurable": {
            "order_id": "",
            "thread_id": thread_id,
        }
    }

    user_conversations[user_id]["history"].append(f"\U0001F9D1\u200D\U0001F4BB You: {user_message}")

    try:
        events = part_1_graph.stream(
            {"messages": [("user", user_message)]}, config, stream_mode="values"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI response: {str(e)}")

    last_assistant_response = ""
    raw_events = list(events)
    for event in raw_events:
        if "messages" in event:
            for message in event["messages"]:
                if hasattr(message, "content") and "AIMessage" in str(type(message)):
                    content = message.content
                    if isinstance(content, dict) and "text" in content:
                        content = content["text"]
                    elif isinstance(content, list):
                        content = " ".join(str(part) for part in content)
                    elif isinstance(content, str):
                        last_assistant_response = content

    return {"response": last_assistant_response}


@app.get("/")
def index():
    # Serve the index.html file from the current directory
    return FileResponse("index.html", media_type="text/html")