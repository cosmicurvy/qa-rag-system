from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_pdfs(file_path):
    """Loads all PDF files from the specified directory and returns them as a list of documents"""

    loader = DirectoryLoader(path = file_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
