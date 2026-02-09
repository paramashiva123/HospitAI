import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime

current_dir = Path(__file__).parent
src_path = str(current_dir / "rag_crew" / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from rag_crew.crew import RagCrew

st.set_page_config(page_title="HospitAI RAG", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "crew" not in st.session_state:
    with st.spinner("🏥 Initializing Hospital Knowledge Base..."):
        st.session_state.crew = RagCrew()

with st.sidebar:
    st.title("🏥 System Monitor")
    st.info("Agentic RAG active across Doctor, Patient, and Financial domains.")
    
    col1, col2 = st.columns(2)
    col1.metric("Queries", len([m for m in st.session_state.messages if m["role"] == "user"]))
    col2.metric("Status", "Online", delta_color="normal")
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Medical RAG Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about treatments, finances, or patient history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.status("🔍 Agent searching medical records...")
        
        try:
            inputs = {"query": prompt}
            result = st.session_state.crew.crew().kickoff(inputs=inputs)
            
            full_response = result.raw if hasattr(result, 'raw') else str(result)
            
            status_placeholder.empty()
            st.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")