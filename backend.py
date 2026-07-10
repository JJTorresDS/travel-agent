import os
from dotenv import load_dotenv

load_dotenv(override=True)

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")

# ================================ Model ================================

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=GEMINI_API_KEY)

# ================================ State ================================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

# ================================ Helpers ================================

def extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)

# ================================ Flight Agent ================================

def flight_agent(state: TravelState) -> TravelState:
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight results fetched successfully")],
        "llm_calls": state.get("llm_calls", 0)+ 1
        }

# ================================ Hotel Agent ================================

def hotel_agent(state: TravelState) -> TravelState:
    query = f"Best hotels in {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel results fetched successfully")],
        "llm_calls": state.get("llm_calls", 0)+ 1
        }

# ================================ Itinerary Agent ================================

def itinerary_agent(state: TravelState):
    prompt = f"""
    Create a complete travel itinerary.

    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}

    Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": extract_text_content(response.content),
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ================================ Final Agent ================================

def final_agent(state: TravelState):
    final_prompt = f"""
    Generate the final travel response for the user.

    User Request:
    {state['user_query']}

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}

    Format the final answer beautifully using these sections:

    1. Trip Summary
    2. Flight Information
    3. Hotel Suggestions
    4. Day-by-Day Itinerary
    5. Estimated Budget
    6. Final Recommendations

    Important:
    - Be clear and practical.
    - Mention that live flight API may not provide ticket prices if pricing is unavailable.
    - Keep the response useful for real travel planning.
    """

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# ================================ Graph ================================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

# ================================ Checkpoint ================================
_conn = psycopg.connect(
    DATABASE_URL
    , autocommit=True
    , row_factory=dict_row
    )


checkpointer = PostgresSaver(_conn)
checkpointer.setup()


travel_graph = graph.compile(checkpointer=checkpointer)

# ================================ FastAPI Function ================================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"
    
    config = {
        "configurable": {"thread_id": thread_id}
        }
    
    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        }
        , config=config
        )
    
    final_answer = extract_text_content(result["messages"][-1].content)

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result["flight_results"],
        "hotel_results": result["hotel_results"],
        "itinerary": extract_text_content(result["itinerary"]),
        "llm_calls": result["llm_calls"]
        }