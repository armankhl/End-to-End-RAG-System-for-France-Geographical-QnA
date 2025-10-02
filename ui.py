# ui.py
import streamlit as st
import requests

st.title("France Geography RAG Chatbot")

query = st.text_input("Ask a question about the geography of France:")

if st.button("Generate Answer"):
    if query:
        with st.spinner("Retrieving context and generating answer..."):
            response = requests.post(
                "http://127.0.0.1:8000/generate",
                json={"query": query, "top_k": 3}
            )
            if response.status_code == 200:
                data = response.json()
                st.success("Answer:")
                st.write(data['answer'])
                with st.expander("See retrieved context"):
                    st.json(data['retrieved_chunks_for_context'])
            else:
                st.error(f"An error occurred: {response.text}")

# streamlit run ui.py