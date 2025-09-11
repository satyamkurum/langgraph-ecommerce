# src/graph/support_graph.py
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools.faq import faq_lookup, get_faq_retriever
from src.tools.orders import track_order
from src.tools.recommender import recommend_products

from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file


# 1) Define the State: messages list + an 'intent' field (used to route)
class State(MessagesState):
    # messages (inherited). We add a routing key:
    intent: str  # one of: "faq", "order", "recommendation", "fallback"

# 2) Initialize an LLM for fallback classification & fallback answers
# Make sure GOOGLE_API_KEY is set in env or pass google_api_key param

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=os.getenv("GOOGLE_API_KEY"))


# Helper: extract latest user text
def _last_user_text(state: State) -> str:
    msgs = state.get("messages", [])
    # messages can be message objects or dicts; try robust extraction
    if not msgs:
        return ""
    last = msgs[-1]
    # support both object and dict formats
    txt = ""
    if hasattr(last, "content"):
        txt = getattr(last, "content")
    elif isinstance(last, dict):
        txt = last.get("content", "")
    else:
        txt = str(last)
    if isinstance(txt, list):  # some LLM returns list-of-parts; flatten
        txt = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in txt])
    return txt

# 3) Nodes
def intent_node(state: State):
    text = _last_user_text(state).lower()
    # simple heuristics
    if any(tok in text for tok in ("order", "track", "where is my order", "order id")):
        return {"intent": "order"}
    if any(tok in text for tok in ("return", "refund", "cancel", "warranty", "shipping")):
        return {"intent": "faq"}
    if any(tok in text for tok in ("recommend", "suggest", "which phone", "what should i buy", "recommendation")):
        return {"intent": "recommendation"}
    # fallback to LLM classifier
    prompt = (
        "Classify the user's intent into exactly one of the labels: "
        "faq, order, recommendation, fallback.\n\n"
        f"User text:\n\"\"\"{text}\"\"\"\n\n"
        "Reply only with one word label."
    )
    resp = llm.invoke(prompt)
    label = resp.content.strip().lower().split()[0]
    if label not in ("faq", "order", "recommendation", "fallback"):
        label = "fallback"
    return {"intent": label}

def faq_node(state: State):
    text = _last_user_text(state)
    answer = faq_lookup(text)
    return {"messages": [AIMessage(answer)]}

def order_node(state: State):
    text = _last_user_text(state)
    # basic order id extraction (digits). more robust parsing can be added.
    import re
    m = re.search(r"\b(\d{4,10})\b", text)
    if not m:
        return {"messages": [AIMessage("Please provide your order ID (e.g. 12345) so I can check the status.")] }
    order_id = m.group(1)
    res = track_order(order_id)
    return {"messages": [AIMessage(res)]}

def recommendation_node(state: State):
    text = _last_user_text(state)
    items = recommend_products(text, top_k=3)
    if not items:
        return {"messages": [AIMessage("I couldn't find product matches. Could you share what you want (budget, category)?")]}
    lines = ["Here are product suggestions:"] + [f"- {p['title']}: {p.get('description','')}" for p in items]
    return {"messages": [AIMessage("\n".join(lines))]}

def fallback_node(state: State):
    # Use the LLM to answer using the conversation history
    # Prepare a simple prompt with last user message
    text = _last_user_text(state)
    prompt = f"You are a helpful customer support assistant. Answer concisely.\n\nUser: {text}\nAssistant:"
    resp = llm.invoke(prompt)
    return {"messages": [AIMessage(resp.content)]}

# 4) Build graph
def build_graph():
    builder = StateGraph(State)
    builder.add_node("intent", intent_node)
    builder.add_node("faq", faq_node)
    builder.add_node("order", order_node)
    builder.add_node("recommendation", recommendation_node)
    builder.add_node("fallback", fallback_node)

    # entry point
    builder.add_edge(START, "intent")
    # after intent runs, route based on intent field in state
    def router(s: State):
        return s.get("intent", "fallback")
    builder.add_conditional_edges("intent", router, {"faq": "faq", "order": "order", "recommendation": "recommendation", "fallback": "fallback"})

    graph = builder.compile()
    return graph

# 5) Example invocation helper
def invoke_graph(graph, user_text: str):
    # Accept either dict messages or langchain messages
    result = graph.invoke({"messages": [HumanMessage(content=user_text)]})
    # return last AI message content
    out_messages = result.get("messages", [])
    if not out_messages:
        return result
    last = out_messages[-1]
    # `last` might be AIMessage object or dict
    if hasattr(last, "content"):
        return last.content
    if isinstance(last, dict):
        return last.get("content", "")
    return str(last)
