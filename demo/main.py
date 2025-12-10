"""Streamlit app for RAG demo.

Start from project root with :
```bash
PYTHONPATH=. streamlit run demo/main.py
```
"""

import requests
import streamlit as st

from demo.ping import display_message_if_ping_fails

# Map source types to display labels
SOURCE_TYPE_LABELS = {
    "paths": "Path",
    "components": "Schema",
    "webhooks": "Webhook",
}


def render_sources(sources: list[dict]) -> None:
    """Render sources expander with formatted source info.

    Args:
        sources: List of source dicts with api_name, source_type, and resource_name.
    """
    if not sources:
        return

    with st.expander(f"📄 Sources ({len(sources)} documents)"):
        for source in sources:
            api_name = source.get("api_name", "unknown")
            source_type = source.get("source_type", "unknown")
            resource_name = source.get("resource_name", "unknown")

            spec_url = f"https://api.eu1.stackone.com/oas/{api_name}.json"
            type_label = SOURCE_TYPE_LABELS.get(source_type, source_type.title())

            st.markdown(
                f"[{api_name}.json]({spec_url}) - {type_label}: `{resource_name}`"
            )

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

    # Configuration selector
    st.divider()
    st.subheader("⚙️ Configuration")
    try:
        # Fetch available configs and current config
        configs_response = requests.get("http://localhost/configs", timeout=2)
        config_response = requests.get("http://localhost/config", timeout=2)

        if configs_response.status_code == 200 and config_response.status_code == 200:
            configs_data = configs_response.json()
            config_data = config_response.json()

            available_configs = configs_data.get("configs", [])
            current_config = configs_data.get("current", "")

            if available_configs:
                # Config selector with descriptive labels
                config_labels = {
                    "c0": "Baseline",
                    "c1": "Smart Chunking",
                    "c2": "Hybrid Search",
                    "c3": "Metadata Filtering",
                    "c4": "Reranking",
                    "c5": "Full System",
                }
                selected_config = st.selectbox(
                    "Select Configuration",
                    options=available_configs,
                    index=available_configs.index(current_config) if current_config in available_configs else 0,
                    format_func=lambda x: f"{x.upper()} - {config_labels.get(x, x)}",
                )

                # Switch config if changed (this also auto-selects the matching collection)
                if selected_config != current_config:
                    requests.post(
                        f"http://localhost/configs/{selected_config}/select", timeout=2
                    )
                    st.rerun()

                # Show current config details
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
                st.warning("No configurations available")
        else:
            st.warning("Could not fetch config from API")
    except Exception as e:
        st.warning(f"API not available: {e}")

    # Show search indices status
    st.divider()

    # Check if hybrid search is enabled (config_data is set above)
    use_hybrid = False
    try:
        use_hybrid = config_data.get("use_hybrid_search", False)
    except NameError:
        pass

    if use_hybrid:
        st.subheader("📚 Search Indices")
    else:
        st.subheader("📚 Vector Store")

    try:
        coll_response = requests.get("http://localhost/collections", timeout=2)
        if coll_response.status_code == 200:
            coll_data = coll_response.json()
            current_collection = coll_data.get("current", "")

            if current_collection:
                # Vector Store status
                collections = coll_data.get("collections", [])
                vector_exists = current_collection in collections

                if use_hybrid:
                    # Show both indices
                    col1, col2 = st.columns(2)
                    with col1:
                        if vector_exists:
                            st.success(f"Vector Store: `{current_collection}`")
                        else:
                            st.warning(f"Vector Store: `{current_collection}` (not loaded)")

                    # Check BM25 index status
                    with col2:
                        try:
                            bm25_response = requests.get("http://localhost/bm25/status", timeout=2)
                            if bm25_response.status_code == 200:
                                bm25_data = bm25_response.json()
                                if bm25_data.get("has_index"):
                                    doc_count = bm25_data.get("document_count", 0)
                                    st.success(f"BM25 Index: {doc_count} docs")
                                else:
                                    st.warning("BM25 Index: not loaded")
                            else:
                                st.warning("BM25 Index: status unavailable")
                        except Exception:
                            st.warning("BM25 Index: status unavailable")
                else:
                    # Vector-only mode
                    if vector_exists:
                        st.info(f"**Collection:** `{current_collection}`")
                        st.caption("Collection exists and is active")
                    else:
                        st.warning(f"Collection `{current_collection}` not found. Load documents first.")
            else:
                st.info("No collection selected")
        else:
            st.warning("Could not fetch collections from API")
    except Exception as e:
        st.warning(f"Could not fetch collections: {e}")

st.title("RAG Example 🤖")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"},
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Display sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

if prompt := st.chat_input("Which path gives me the candidate list?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        try:
            response = requests.post("http://localhost/chat", json={"query": prompt})
            response.raise_for_status()
            result = response.json()
            msg = result["message"]
            sources = result.get("sources", [])
        except Exception as e:
            st.error(e)
            st.stop()

    st.empty()

    st.session_state.messages.append({
        "role": "assistant",
        "content": msg,
        "sources": sources,
    })
    with st.chat_message("assistant"):
        st.write(msg)
        if sources:
            render_sources(sources)
