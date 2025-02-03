from flask import Flask, request, jsonify, send_from_directory
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



# Load environment variables
load_dotenv()

# Store user conversations separately
user_conversations = {}

# Endpoint URL
url = os.environ.get("SCL_URL")

# Basic Authentication credentials
username = os.environ.get("SCL_USERNAME")
password = os.environ.get("SCL_PASSWORD")
shipping_url = os.environ.get("SHIPMENT_TRACKING_URL")

# Initialize Flask app
app = Flask(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create or connect to a Pinecone index
index_name = "rag-pinecone-labse"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # LaBSE embeddings are 768-dimensional
        metric="cosine",  # Use cosine similarity for text embeddings
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the index
pinecone_index = pc.Index(index_name)

# Load the LaBSE model
embedding_model = SentenceTransformer('sentence-transformers/LaBSE')

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
def product_information() -> str:
    """Retrieve pricing information for the product."""
    print("product_information")
    products = pd.read_csv("product_information1.csv")
    return products.to_string() + "\n\n ."

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


# Define the State class
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

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

# Initialize the LLM and tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=1)
web_search_tool = TavilySearchResults(k=3, include_domains=["https://sanaexpert.es/"])
part_1_tools = [web_search_tool, order_information, product_information, knowledgebase_sanaexpert, escalate_to_human]

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
6. Use tools ONLY when specific data is needed.
7. Maintain professional yet approachable tone.
8. Clarify ambiguous requests before acting.
9. Keep your response very brief and concise and ask one thing at a time.
10. Use tools information only in the background and don't tell it to the customer. If you can't find any info from knowledgebase and other sources you can try web_search_tool.
11. In Case you are not sure about answer just ask customer for his name and email if not provided before. and then tell the user that you are escalating the ticket to human representation and then call escalate_to_human tool."""),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

# Build the assistant runnable
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

# Build the graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)

# In-memory storage for user conversations
user_conversations = {}

def fetch_link_metadata(url):
    """Fetch the title and featured image of a webpage."""
    print("Fetching metadata for:", url)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract Open Graph metadata
        title = soup.find("meta", property="og:title") or soup.find("title")
        image = soup.find("meta", property="og:image")

        title_text = title["content"] if title and title.has_attr("content") else title.get_text() if title else url
        image_url = image["content"] if image and image.has_attr("content") else None

        return {"title": title_text, "image": image_url, "url": url}
    except Exception as e:
        print("Error fetching metadata:", e)
        return None


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id')  # Unique identifier for each user
    user_message = data.get('message')

    if not user_id or not user_message:
        return jsonify({"error": "Both user_id and message are required"}), 400

    # Retrieve or create a new conversation for the user
    if user_id not in user_conversations:
        user_conversations[user_id] = {
            "thread_id": str(uuid.uuid4()),
            "history": []
        }

    url_regex = r'(https?:\/\/[^\s.,!?()"\'\]]+)'
    urls = re.findall(url_regex, user_message)

    link_previews = []
    for url in urls:
        print("URL detected:", url)
        metadata = fetch_link_metadata(url)
        if metadata:
            link_previews.append(metadata)

    thread_id = user_conversations[user_id]["thread_id"]
    config = {
        "configurable": {
            "order_id": "",
            "thread_id": thread_id,
        }
    }

    # Append user message to conversation history
    user_conversations[user_id]["history"].append(f"\U0001F9D1\u200D\U0001F4BB You: {user_message}")

    # Fetch response from assistant
    try:
        events = part_1_graph.stream(
            {"messages": [("user", user_message)]}, config, stream_mode="values"
        )
    except Exception as e:
        return jsonify({"error": f"Failed to fetch AI response: {str(e)}"}), 500

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
                        if "<function=" in content:
                            content = "Your issue has been escalated to a human representative for further assistance."
                    else:
                        continue
                    last_assistant_response = content.strip()

    assistant_response = last_assistant_response or "[No response]"

    # Append assistant response to conversation history
    user_conversations[user_id]["history"].append(f"\U0001F916 SanaExpert Agent: {assistant_response}")

    return jsonify({
        "response": assistant_response,
        "conversation_history": user_conversations[user_id]["history"],
        "link_previews": link_previews
    })


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)