from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def chat_prompt(question: str, context: str):
    """Retruns a well-structured prompt using the system message, retrieved context and the user's query"""

    message = """You are an expert AI research assistant. Your task is to answer the user's question based strictly on the provided research paper context. Do not use any outside knowledge. 
    CORE INSTRUCTIONS:
    - If the answer cannot be found in the provided context, reply exactly with: "There is no information available for this question." Do not attempt to fabricate an answer.
    - Length & Detail Constraints:
        a) If the user explicitly asks for a detailed explanation, provide a comprehensive response based only on the context strictly to 5-6 sentences.
        b) Otherwise, provide a precise, direct answer limited strictly to 1-3 sentences.
    - Citation Rules: You must back up every claim with an explicit citation naming the source paper. Use the format: (Source: Paper Name, Page X/Section Y) if available in the context.
    Context: {context}"""

    chat_template = ChatPromptTemplate.from_messages(messages=[('system', message),
                                                                ('human', '{question}')])
    

    prompt = chat_template.format_messages(
        context = context,
        question = question)
    
    return prompt


def call_llm(prompt: list):
    """Takes a prompt and returns a response from the llm"""
    llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash", max_tokens=1000)
    response = llm.invoke(prompt)
        
    return response.content[0]['text']