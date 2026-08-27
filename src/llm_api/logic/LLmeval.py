import json
from .hybridSearch import hybrid_search_with_rerank
from openai import OpenAI
from ..models.Credintials import qwen

from datasets import Dataset
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric
from deepeval.models import OllamaModel,OpenAIModel
from deepeval.models import GeminiModel, AnthropicModel
from deepeval.evaluate.evaluate import CacheConfig  # import cache config
from time import sleep
client = OpenAI(base_url="http://localhost:11434/v1/", api_key='ollama')

 

# 2. Test your Anthropic Key
anthropic_model = AnthropicModel(
    model="claude-3-haiku-20240307", 
    api_key=api-key,
)

faithfulness_metric = FaithfulnessMetric(threshold=0.7,model=anthropic_model)
answer_relevance_metric = AnswerRelevancyMetric(threshold=0.7,model=anthropic_model)
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
                input=q,                        
                actual_output=llm,         
                retrieval_context=context_rag,         
                expected_output=expected_answer             
            )
        testing_case.append(test_case)
        print(f'appended item : {i}')
        i+=1

        sleep(6)
    evaluate(testing_case, metrics=[faithfulness_metric, answer_relevance_metric],cache_config=CacheConfig(write_cache=False, use_cache=False))




if __name__ == "__main__":
    eval()