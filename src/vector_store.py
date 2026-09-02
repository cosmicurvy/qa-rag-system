from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
    
def vector_store(chunks: list, persist_dir='./chroma_db'):
    """Stores embeddings of the chunked documents in a Chroma vector database"""

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, 
                                            encode_kwargs={'normalize_embeddings': True})

    vectorstore = Chroma.from_documents(embedding=embedding_model, documents = chunks, persist_directory=persist_dir)

    return vectorstore



