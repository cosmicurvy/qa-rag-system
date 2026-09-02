
def retrieved_topK(query: str, vector_s, k = 3):
    """"Retrieves Top K chunks from the knowledge base"""
    retriever = vector_s.as_retriever(search_kwargs={'k': k})
    retrieved_docs = retriever.invoke(query)

    context = ""

    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get('source')
        context += f"Document {i+1} (Source: {source})\n"
        context += doc.page_content + "\n\n"

    return context
