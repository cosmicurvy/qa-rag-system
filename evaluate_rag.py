from deepeval import evaluate
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.evaluate import AsyncConfig, ErrorConfig

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd

from src.generation import call_llm, chat_prompt
from src.retriever import retrieved_topK

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300" # gives 5 minutes per test case

local_judge = OllamaModel(model="llama3.2")

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'},
                                            encode_kwargs={'normalize_embeddings': True})

vector_store = Chroma(persist_directory='./chroma_db', embedding_function=embeddings_model)

# questions to evaluate
questions = [
   "Why does the Transformer use multi-head attention instead of a single attention mechanism?",
    "How does BERT use bidirectional context during pre-training?",
    "What are the two pre-training tasks used to train BERT?"
    ]

print("Running test cases...")
# creating DeepEval test cases
test_cases = []

for question in questions:
    docs = vector_store.similarity_search(question, k=3) # returns a list, it's for deepeval tests
    retrieved_chunks = [doc.page_content for doc in docs]

    retrieved_docs = retrieved_topK(query=question, vector_s=vector_store, k=3) # returns a string
    prompt = chat_prompt(question=question, context=retrieved_docs) 
    answer = call_llm(prompt) 


    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieved_chunks
        )

    test_cases.append(test_case)

print("Defining metrics...")
# defining metrics
metrics = [AnswerRelevancyMetric(threshold=0.7, include_reason=True, model=local_judge),
           FaithfulnessMetric(threshold=0.7, include_reason=True, model=local_judge),
           ContextualRelevancyMetric(threshold=0.7, include_reason=True, model=local_judge)
           ]

async_configuration = AsyncConfig(run_async=True,
                                  max_concurrent=1,   # run 1 test case at a time 
                                  throttle_value=2    # rest 2 seconds between test cases
                                  )

error_configuration = ErrorConfig(
    ignore_errors=True  # Skip failing cases without crashing the script
    )


print("Evaluating...")
# running evaluation
results = evaluate(test_cases=test_cases, metrics=metrics, async_config= async_configuration, error_config=error_configuration)


# convert the results object into a CSV file
data = [] 
for test_run in results.test_results:
    row = {
        'Input' : test_run.input,
        "Actual Output": test_run.actual_output,
        "Success": test_run.success
    }

    for metric_result in test_run.metrics_data:
        metric_name = metric_result.name
        row[f"{metric_name} Score"] = metric_result.score
        row[f"{metric_name} Reason"] = metric_result.reason
        
    data.append(row)

df = pd.DataFrame(data)
df.to_csv("evaluation_results/rag_evaluation_results.csv", index=False)
print("Results successfully saved!!!")