from src.splitter import split_docs
from src.generation import chat_prompt
from backend import app
from langchain_core.documents import Document
from fastapi.testclient import TestClient

client = TestClient(app)

# Tests FastAPI endpoint
def test_ask_endpoint():
    """Tests whether it returns 422 status code for an empty question"""
    response = client.post('/ask', json={})
    assert response.status_code == 422


# Tests RAG components
def test_split_docs():
    """Test if splitter.py correctly splits the documents into chunks."""
    long_text = "abc. " * 10000
    mock_document = Document(page_content=long_text, metadata={'source': 'test.pdf'})

    chunks = split_docs([mock_document])

    # assertions
    assert len(chunks) > 1 
    assert len(chunks[0].page_content) <= 800 
    assert chunks[0].metadata['source'] == 'test.pdf'


def test_chat_prompt_generation():
    """Test if generation.py correctly formats the human and system prompts"""
    test_question = "How does BERT work?"
    test_context = "BERT is deeply bidirectional"

    prompt_list = chat_prompt(question=test_question, context=test_context)

    system_msg = prompt_list[0].content
    human_msg = prompt_list[1].content

    # assertions
    assert len(prompt_list) == 2

    assert test_context in system_msg
    assert test_question in human_msg

    assert "You are an expert AI research assistant." in system_msg
