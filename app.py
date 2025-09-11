# app.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.graph.support_graph import build_graph, invoke_graph

# Build the LangGraph once (reuse for the whole session)
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Customer Support Bot", page_icon="💬", layout="wide")
st.title("Customer Support Assistant")

# Sidebar instructions
with st.sidebar:
    st.header("About")
    st.write(
        """
        This is an AI-powered **customer support assistant** built with:
        - **LangGraph** (stateful workflow graph)
        - **LangChain** tools (FAQ, order tracking, recommendations)
        - **Google Gemini** (LLM for intent detection and fallback answers)

        You can ask about:
        - Order tracking (e.g., "Track order 12345")
        - FAQs (e.g., "What is your return policy?")
        - Product recommendations (e.g., "Suggest me a good laptop under 50000")
        """
    )

# Display chat messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.write(user_input)

    # Get response from backend graph
    response_text = invoke_graph(st.session_state.graph, user_input)

    st.session_state.chat_history.append(AIMessage(content=response_text))

    with st.chat_message("assistant"):
        st.write(response_text)
