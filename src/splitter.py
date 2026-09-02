from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(docs: list):
    """Splits a list of documents into fixed-sized chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". "])

    chunks = splitter.split_documents(docs)

    return chunks
