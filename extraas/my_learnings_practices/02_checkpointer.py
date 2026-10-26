# creating a memory check pointer
import os
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,AIMessage, HumanMessage
from pydantic import BaseModel
from typing import Annotated, List
from dotenv import load_dotenv

load_dotenv()

