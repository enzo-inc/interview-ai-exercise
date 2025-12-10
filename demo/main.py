"""Streamlit app for RAG demo.

Start from project root with :
```bash
PYTHONPATH=. streamlit run demo/main.py
```
"""

import requests
import streamlit as st

from demo.ping import display_message_if_ping_fails

st.set_page_config(
    "RAG Example",
)

if "session" not in st.session_state:
    st.session_state.session = {}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"},
    ]


with st.sidebar:
    display_message_if_ping_fails()
    
    # Display current configuration
    st.divider()
    st.subheader("⚙️ Configuration")
    try:
        config_response = requests.get("http://localhost/config", timeout=2)
        if config_response.status_code == 200:
            config_data = config_response.json()
            st.info(f"**Config:** `{config_data['name']}`")
            st.caption(config_data["description"])
            
            # Show enabled features
            features = []
            if config_data.get("use_smart_chunking"):
                features.append("Smart Chunking")
            if config_data.get("use_hybrid_search"):
                features.append("Hybrid Search")
            if config_data.get("use_metadata_filtering"):
                features.append("Metadata Filtering")
            if config_data.get("use_reranking"):
                features.append("Reranking")
            if config_data.get("use_unknown_detection"):
                features.append("Unknown Detection")
            
            if features:
                st.write("**Enabled:** " + ", ".join(features))
            else:
                st.write("**Mode:** Baseline")
        else:
            st.warning("Could not fetch config from API")
    except Exception as e:
        st.warning(f"API not available: {e}")
    
    st.divider()
    st.warning(
        "⚠️ **Note:** Ensure that the vector/lexical indices have been "
        "manually created prior to using this demo. Run `/load` endpoint "
        "or use the data loader to populate the vector store."
    )

st.title("RAG Example 🤖")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"},
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Which path gives me the candidate list?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = ""

    with st.spinner("Thinking..."):
        try:
            response = requests.post("http://localhost/chat", json={"query": prompt})
            response.raise_for_status()
            result = response.json()
            msg = result["message"]
        except Exception as e:
            st.error(e)
            st.stop()

    st.empty()

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
