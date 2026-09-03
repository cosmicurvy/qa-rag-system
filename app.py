import streamlit as st
import requests
import os


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/ask")

st.set_page_config(page_title="Question-Answering RAG System")
st.title("Question-Answering RAG System")
st.markdown("Ask questions based on the provided research papers- *AttenionIsAllYouNeed*, *BERT* ")
st.divider()

with st.form(key='qa_form'):
    question_input = st.text_area("What would you like to ask?", height=100, placeholder='e.g- What are transformers?')
    submit= st.form_submit_button(label = "Ask Question")


if submit:
    if not question_input.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Scanning the research papers and generating the answers..."):
            try:
                payload = {'question': question_input}
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # if the server encounters error then raises an HTTPError Exception, otherwise does nothing.          
                data = response.json()

                answer = data.get("answer", "Error: no answer returned")

                st.subheader("Answer")
                st.info(answer)

            except requests.exceptions.RequestException as e:
                st.error(f"Backend connection failed. Ensure fastapi is running on port 8000. Error: {e}")


