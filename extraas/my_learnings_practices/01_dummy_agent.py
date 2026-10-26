import os
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from typing import List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("INMETECH_API_KEY")

if not api_key:
    raise ValueError("INMETECH_API_KEY is missing from your .env file.")

llm = ChatOpenAI(
    model="gpt-4o",
    base_url="https://api.inmetech.com/api/v1",
    api_key=api_key,
)

# state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# nodes
def chat_node(state: AgentState)-> dict:
    
    conversation_history = state["messages"]
    response = llm.invoke(conversation_history)
    
    return {
        "messages": response
    }

# builder
agent_graph = StateGraph(AgentState)

# connecting nodes and edges
agent_graph.add_node("chat_node",chat_node)

agent_graph.add_edge(START,"chat_node")
agent_graph.add_edge("chat_node", END)

# compile the graph
agent = agent_graph.compile()


# human message
message1 = HumanMessage(content="What is AI ?")

# turn 1 of the conversations
turn1_state = agent.invoke({
    "messages": message1
})

# print the results
print("---------Turn 1 results----------")
print(turn1_state)
print("-"*20)


# message 2
message2  = HumanMessage(content="who is the PM of Bangladesh ?")

# turn 02
turn2_state = agent.invoke({
    "messages": turn1_state["messages"]+[message2]
})
# current message with previous message

# print the results
print("\n\n---------Turn 2 results----------")
print(turn2_state)
print("-"*20)



