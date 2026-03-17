import streamlit as st
import asyncio
import yaml
from pathlib import Path
from core.orchestrator import RAGOrchestrator

# --- App Configuration ---
st.set_page_config(
    page_title="Elite Azure RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("Elite Azure RAG Assistant 🤖")
st.markdown("Ask anything! Queries are simultaneously routed across massive Azure Search indexes and open-source intelligence via MCP.")

# --- Load YAML Configurations ---
@st.cache_data
def load_index_configs(file_path="indexes.yaml"):
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            return data.get("indexes", [])
    except Exception as e:
        st.error(f"Error loading index configurations: {e}")
        return []

INDEXES = load_index_configs()

# Placeholder MCP configurations. You can move these to a yaml as well.
MCP_SERVERS = [
    {
        "name": "Local-Intelligence",
        "url": "http://localhost:8000/mcp",
        "description": "Internal tools and APIs"
    }
]

# --- Initialize Memory / Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Helper to handle async orchestrator in a sync app ---
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

# --- Chat Interface Execution ---
if prompt := st.chat_input("What would you like to know?"):
    # 1. Add User Input to State
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Adding a visual indication of routing logic processing 
        with st.status("🧠 Analyzing query and routing to appropriate indexes..."):
            st.write(f"Routing logic initialized for **{len(INDEXES)}** Azure Search indexes and **{len(MCP_SERVERS)}** MCP servers.")
            
            # Execute Pipeline
            orchestrator = RAGOrchestrator()
            loop = get_or_create_eventloop()
            
            try:
                # Retrieve the full conversation context (memory) to include in the query
                history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
                contextual_prompt = f"Previous Context:\n{history}\n\nCurrent Question: {prompt}"
                
                # Mock response for testing UI without full Azure credentials / OpenAI installation
                import time
                time.sleep(1.5) # Simulate processing time
                mock_sources = "\n\n**Sources:**\n- [HR Policy V1](#) (hr-docs-index)\n- [IT Resolution 4022](#) (it-support-index)"
                answer = f"This is a simulated response from the Elite Azure RAG Pipeline for your query: '{prompt}'.\n\nThe routing logic successfully checked {len(INDEXES)} Azure Search indexes and integrated results. {mock_sources}"
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                st.error(f"Pipeline Execution Failed: {e}")
                st.expander("Show Traceback").code(error_trace)
                answer = "An error occurred during process orchestration."
            
        message_placeholder.markdown(answer)
        
    # 3. Add Assistant Response to State
    st.session_state.messages.append({"role": "assistant", "content": answer})
