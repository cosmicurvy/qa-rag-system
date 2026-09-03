# Question Answering RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system for question answering over two influential NLP research papers:

* *Attention Is All You Need* — Vaswani et al.
* *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — Devlin et al.

The system retrieves relevant passages from the papers and uses them as context to generate grounded answers. The application is built as a modular Python system with an API backend, interactive frontend, automated testing, RAG evaluation, Docker, and CI/CD.

## Overview

The system enables users to ask questions about the research papers and receive answers based on retrieved document context rather than relying solely on the language model's internal knowledge.

**RAG Flow:**

`Research Papers → Chunking → Embeddings → ChromaDB → Retrieval → Context → Generation → Answer`

## Evaluation

RAG performance was evaluated using **DeepEval** across three questions using:

* **Answer Relevancy** — relevance of the generated answer to the question.
* **Faithfulness** — whether the generated answer is supported by the retrieved context.
* **Contextual Relevancy** — relevance of the retrieved context to the question.

### Results

| Metric               | Average Score |
| -------------------- | ------------: |
| Answer Relevancy     |      **1.00** |
| Faithfulness         |      **0.87** |
| Contextual Relevancy |      **0.70** |

The results indicate that the generated answers were **highly relevant to the questions**, while faithfulness was also strong overall. Contextual relevancy was comparatively lower, indicating an opportunity to improve the retrieval component by reducing irrelevant retrieved passages.

### Evaluation Examples

| Question                                           | Answer Relevancy | Faithfulness | Contextual Relevancy |
| -------------------------------------------------- | ---------------: | -----------: | -------------------: |
| Why does the Transformer use multi-head attention? |             1.00 |         0.60 |                 0.91 |
| How does BERT use bidirectional context?           |             1.00 |         1.00 |                 0.70 |
| What are BERT's two pre-training tasks?            |             1.00 |         1.00 |                 0.50 |

## Application

The system provides a **Streamlit** interface for asking questions and a **FastAPI** backend for serving the RAG pipeline.

`Streamlit → FastAPI → RAG Pipeline → ChromaDB → LLM → Response`

Automated tests using **Pytest** validate the RAG components and FastAPI endpoint.

## Engineering

* Modular Python implementation
* Automated RAG evaluation with DeepEval
* Automated testing with Pytest
* Dockerized application
* CI/CD pipeline using GitHub Actions
* FastAPI REST API
* Streamlit user interface

## Tech Stack

**Python | LangChain | Hugging Face | ChromaDB | DeepEval | FastAPI | Streamlit | Pytest | Docker | GitHub Actions | Git**

## Example Questions

* Why does the Transformer use multi-head attention?
* What is the purpose of positional encoding?
* How does BERT use bidirectional context?

## Key Takeaways

This project demonstrates practical experience in **RAG development, semantic retrieval, vector databases, LLM evaluation, API development, automated testing, containerization, and CI/CD**.

