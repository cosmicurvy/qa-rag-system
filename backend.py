from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from src.loader import load_pdfs
from src.splitter import split_docs
from src.vector_store import vector_store
from src.retriever import retrieved_topK
from src.generation import chat_prompt, call_llm

DATA_DIR = "./data"
VECTOR_DB_DIR = "./chroma_db"
vectorstore = None

app = FastAPI(title="RAG Backend API")

class QueryRequest(BaseModel):
    question: str

@app.on_event('startup')
async def startup_event():
    """Initializes the vectorstore on startup"""
    global vectorstore

    if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
        print("Vector database is not found. Initializing and building...")

        if not os.path.exists("DATA_DIR"):
            print("Data directory does not exist.")
        
        docs = load_pdfs(DATA_DIR)
        if not docs:
            print(f"PDF files are not found in the {DATA_DIR}")

        chunks = split_docs(docs)

        vectorstore = vector_store(chunks, persist_dir=VECTOR_DB_DIR)
        print("Vector Database built sucessfully!")

    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, 
                                            encode_kwargs={'normalize_embeddings': True})

        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_model)
        print("Vector database loaded from disk...")


@app.post('/ask')
async def ask_question(request : QueryRequest):
    """Endpoint to process the query through the rag pipeline"""

    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vector store is not initialized.")
    
    try:
        context = retrieved_topK(query=request.question, vector_s= vectorstore, k = 3)
        prompt = chat_prompt(request.question, context)
        response = call_llm(prompt)

        return {
            'answer' : response
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))