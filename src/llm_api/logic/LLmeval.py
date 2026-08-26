import json
from .hybridSearch import hybrid_search_with_rerank
from openai import OpenAI
from ..models.Credintials import qwen

from datasets import Dataset
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric
from deepeval.models import OllamaModel,OpenAIModel
from deepeval.evaluate.evaluate import CacheConfig  # import cache config
from time import sleep
client = OpenAI(base_url="http://localhost:11434/v1/", api_key='ollama')

local_model = OpenAIModel(
    model=qwen,
    base_url="http://localhost:11434/v1/",
    api_key='ollama'
)


faithfulness_metric = FaithfulnessMetric(threshold=0.7,model=local_model,async_mode)
answer_relevance_metric = AnswerRelevancyMetric(threshold=0.7,model=local_model, )
def load_qs():
    path = 'questions.json'
    with open(path) as f:
        return json.load(f)


def llm_answers(context , query:str):
    res = client.responses.create(
        model=qwen,
        input=f"""Answer the question using ONLY the information in the context below.
If the context doesn't contain the answer, say "I don't know based on the given context."
 
Context:
{context}
 
Question: {query}
Answer (1-2 sentences):"""
    )
    return res.output_text



def eval():

    testing_case=[]
    questions_json = load_qs()
    i=0
    rate_limit=0
    for item in questions_json:

        q = item['question']
        expected_answer = item['expected_answer']
        context_rag = hybrid_search_with_rerank(q)
        llm= llm_answers(context_rag,query=q)
        test_case = LLMTestCase(
                input=q,                        # The user query
                actual_output=llm,             # Your LLM's response
                retrieval_context=context_rag,          # The database chunks (list of strings)
                expected_output=expected_answer             # Expected answer from your JSON
            )
        testing_case.append(test_case)
        print(f'appended item : {i}')
        i+=1
        if i == 3 :
            evaluate(testing_case, metrics=[faithfulness_metric, answer_relevance_metric],cache_config=CacheConfig(write_cache=False, use_cache=False))
            

            break        
        sleep(6)




if __name__ == "__main__":
    eval()