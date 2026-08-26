from .Generateqs import get_context
from .hybridSearch import hybrid_search_with_rerank
import json

def load_qs():
    path = 'questions.json'
    with open(path) as f: 
        return json.load(f)

def eval():
    context= get_context


