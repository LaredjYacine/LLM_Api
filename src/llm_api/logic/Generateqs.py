from ..models.Connections import createSyncSession ,getsyncConnection
from openai import OpenAI 
from ..models.model import EmbeddedItems
from sqlalchemy import select
from ..models.Credintials import qwen
import json
client = OpenAI(base_url="http://localhost:11434/v1/" , api_key='ollama')

engine = getsyncConnection()
session = createSyncSession(engine)


def get_context():
    try : 
        rows = session.scalars(select(EmbeddedItems).limit(50)).all()
        return rows
    except Exception as e : 
        print('error occured at get context function : ', e )
        return None

    finally: 
        session.close()





def generate_questions(context):
    prompt= f"""You are creating a test question for a search system.
 
Below is a single product review. Write ONE short, natural question a real
user might type to try to find THIS specific review, and ONE short expected
answer that could only come from this review's content. Do not invent facts
not present in the review.
 
Review (id={context.id}):
\"\"\"{context.Combined}\"\"\"
 
Respond ONLY in this exact JSON format, no other text:
{{"question": "...", "expected_answer": "..."}}
"""

    result = client.responses.create(
        model=qwen,
        input=prompt       
    )
    raw = result.output_text
    raw = raw.replace("```json", "").replace("```", "").strip()
    try : 
        parsed=  json.loads(raw)
        return {
            'question' :parsed['question'],
            'expected_answer' : parsed['expected_answer']
            ,'id': context.id 

        }
    except Exception as e : 
        print('error has occured in generate_questions func ' , e )
        return None



if __name__ == '__main__':
    context = get_context()
    qs=[]
    for i, row in enumerate(context) : 
        question = generate_questions(row)
        if question :
            qs.append(question)
            print(f'{i} done out of {len(context)}') 
            
    with open('questions.json', 'w') as f:
        json.dump(qs,f,indent=2)

    print(f"\nGenerated {len(qs)} Q&A pairs -> qa_eval_set.json")
    print("Now spot-check ~10-15 of these by hand and fix any weak ones.")