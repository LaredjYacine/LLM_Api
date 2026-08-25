import bm25s
from openai import OpenAI
import dotenv
import os
import cohere
import Stemmer
from ..models.Connections import createSyncSession, getsyncConnection
from ..models.Credintials import NomicModel
from ..models.model import EmbeddedItems
from sqlalchemy import select 
import numpy as np 
TEST_QUERIES = [
    # --- Keyword-favors-BM25: rare exact terms, little semantic scaffolding ---
    {
        "query": "Robitussin",
        "relevant_ids": ["4"],
    },
    {
        "query": "Ass Kickin brand peanuts",
        "relevant_ids": ["53"],
    },
    {
        "query": "Dolce Gusto machine",
        "relevant_ids": ["31"],
    },

    # --- Semantic-favors-vector: same meaning, no shared keywords ---
    {
        "query": "A candy that reminds someone of a fantasy novel where a child betrays his siblings for something sweet",
        "relevant_ids": ["3"],  # Turkish Delight / C.S. Lewis review, no literal keyword overlap
    },
    {
        "query": "Living abroad and struggling to get a favorite American snack shipped overseas",
        "relevant_ids": ["21", "22", "25"],  # all Twizzlers-overseas reviews, different wording each
    },
    {
        "query": "A spicy snack that turned out not to be spicy at all despite the packaging",
        "relevant_ids": ["54"],
    },

    # --- Multi-relevant / distractor-heavy: many similar reviews, need the RIGHT subset ---
    {
        "query": "Dog food that helped with itching and skin allergies",
        "relevant_ids": ["84", "85", "86", "87", "89", "92", "96", "98", "99"],
    },
    {
        "query": "Instant oatmeal that turned out mushy or bland",
        "relevant_ids": ["46", "48", "51"],
    },
    {
        "query": "Taffy order with too much licorice flavor mixed in",
        "relevant_ids": ["6"],
    },

    # --- Ambiguous / underspecified, real-user phrasing ---
    {
        "query": "candy that had no real flavor, just tasted like plain red sugar",
        "relevant_ids": ["27"],
    },
]
engine = getsyncConnection()
session = createSyncSession(engine=engine)
dotenv.load_dotenv()
cohere_api = os.getenv('cohere')
co = cohere.ClientV2(api_key=cohere_api)
stemmer = Stemmer.Stemmer('english')
client = OpenAI(base_url="http://localhost:11434/v1/",api_key='ollama')

def embed_query(query: str) -> np.ndarray:
    return np.array(
        client.embeddings.create(model=NomicModel, input=query).data[0].embedding
    )
 
 
def vector_only_search(query: str, top_k: int = 5) :

    if query is None:
        raise ValueError('query cannot be empty')
    session = createSyncSession(engine=engine)
    try:
        query_embeddings = embed_query(query)
        responses = session.scalars(
            select(EmbeddedItems)
            .order_by(EmbeddedItems.Embedding.cosine_distance(query_embeddings))
            .limit(top_k)
        ).all()
        return [str(r.id) for r in responses]
    except Exception as e:
        print('error occurred at function vector_only_search', e)
        return []
    finally:
        session.close()











def bm25_Rag(query :str):
    if query is None: 
        raise ValueError('query cannot be empty')
    Hybrid_Data=[]
    try:
        response_corpus= bm25s.BM25.load('bm25Retriever',load_corpus=True)
        corpus = response_corpus.corpus
        query_tokens = bm25s.tokenize(query,stemmer=stemmer,stopwords='en')
        bm , _ = response_corpus.retrieve(query_tokens=query_tokens,corpus=corpus, k=10)
        for docs in bm[0] : 
            Hybrid_Data.append({"id": int(docs['id']), "text": docs['text']})
        query_embeddings =np.array( client.embeddings.create(
            model=NomicModel,
            input = query
        ).data[0].embedding
        )
        responses = session.scalars(select(EmbeddedItems).order_by(EmbeddedItems.Embedding.cosine_distance(query_embeddings)).limit(10)).all()
        for  response in responses : 
            Hybrid_Data.append({"id": response.id, "text": response.Combined})

        return Hybrid_Data
    except Exception as e : 
        print('error occured at function bm25_rag', e)
    finally:
        session.close()
    


def hybrid_search_with_rerank(query : str):

    hybrid_data = bm25_Rag(query)
    seen= set()
    noDups=[]
    for item in hybrid_data: 
        if item['id']  not in seen:
            seen.add(item['id'])
            noDups.append(item) 
    items = [item['text'] for item in noDups]
    if noDups: 


        response = co.rerank(
            model='rerank-v4.0-pro',
            query=query,
            documents=items,
            top_n=5
        )


        res = [res.index for res in response.results]
        top_5 =[noDups[id] for id in res  ]
        top_5_id = [str(item['id']) for item in top_5]
        return top_5_id
 
def recall_at_k(retrieved_ids, relevant_ids) -> float:
    """
    Casts everything to str before comparing, so it doesn't matter whether
    ids arrive as int or str from any given retrieval path.
    """
    if not relevant_ids:
        return None
    retrieved_set = {str(i) for i in retrieved_ids}
    relevant_set = {str(i) for i in relevant_ids}
    found = retrieved_set.intersection(relevant_set)
    return len(found) / len(relevant_set)
 


def run_evaluation():
    baseline_scores = []
    hybrid_scores = []
 
    print(f"{'Query':<55} {'Baseline':>10} {'Hybrid':>10}")
    print("-" * 78)
 
    for item in TEST_QUERIES:
        query = item["query"]
        relevant_ids = item["relevant_ids"]
 
        baseline_retrieved = vector_only_search(query, top_k=5)
        hybrid_retrieved = hybrid_search_with_rerank(query)
 
        baseline_r = recall_at_k(baseline_retrieved, relevant_ids)
        hybrid_r = recall_at_k(hybrid_retrieved, relevant_ids)
 
        baseline_scores.append(baseline_r)
        hybrid_scores.append(hybrid_r)
 
        print(f"{query[:53]:<55} {baseline_r:>9.2%} {hybrid_r:>9.2%}")
 
    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    avg_hybrid = sum(hybrid_scores) / len(hybrid_scores)
    delta = avg_hybrid - avg_baseline
 
    print("-" * 78)
    print(f"{'AVERAGE':<55} {avg_baseline:>9.2%} {avg_hybrid:>9.2%}")
    print()
    print(f"Vector-only baseline Recall@5:  {avg_baseline:.2%}")
    print(f"Hybrid + rerank Recall@5:       {avg_hybrid:.2%}")
    print(f"Delta (improvement):            {delta:+.2%}")
 
    if delta > 0:
        print("\n✅ DoD satisfied: hybrid+rerank measurably improves Recall@5.")
    else:
        print("\n⚠️  No improvement detected — hybrid did not beat vector-only baseline.")
 




if __name__ == "__main__":
    run_evaluation()