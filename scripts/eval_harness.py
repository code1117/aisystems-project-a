"""
Evaluation Harness — Session 1 Starter

This is a SKELETON. During Session 1, we'll build each function
from scratch to create a complete eval pipeline.

Functions to implement:
  1. check_retrieval_hit() — is the expected source in the top-K results?
  2. calculate_mrr() — how high is the first relevant chunk ranked?
  3. judge_faithfulness() — is the answer grounded in the context? (LLM-as-judge)
  4. judge_correctness() — does the answer match the expected answer? (LLM-as-judge)
  5. run_eval() — orchestrate everything and produce a scorecard

Run: python scripts/eval_harness.py
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from rag import ask

load_dotenv()

client = OpenAI()

SCRIPT_DIR = os.path.dirname(__file__)


# =========================================================================
# GOLDEN DATASET
# =========================================================================
# TODO: We'll build this together in Session 1.
# Start with 5 hand-written question-answer-context triples.
# Format:
# {
#     "id": "q01",
#     "query": "What is the standard return window?",
#     "expected_answer": "30 calendar days from delivery date.",
#     "expected_source": "01_return_policy.md",
#     "difficulty": "easy",
#     "category": "returns"
# }
# =========================================================================


def load_golden_dataset():
    """Load the golden dataset from JSON file."""
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        print("No golden_dataset.json found. Create one first!")
        return []
    with open(path) as f:
        return json.loads(f.read())


# =========================================================================
# RETRIEVAL METRICS
# =========================================================================

def check_retrieval_hit(retrieved_chunks, expected_source):
    """
    Is the expected source document in the retrieved chunks?
    Returns True/False.

    TODO: Implement this in Session 1.
    """
    return any(c['doc_name'] == expected_source for c in retrieved_chunks)


def calculate_mrr(retrieved_chunks, expected_source):
    """
    Mean Reciprocal Rank — how high is the first relevant chunk?
    If relevant chunk is at position 1: MRR = 1.0
    If at position 3: MRR = 0.33
    If not found: MRR = 0.0

    TODO: Implement this in Session 1.
    """
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk['doc_name'] == expected_source:
            return 1.0 / rank
    return 0.0


# =========================================================================
# GENERATION METRICS (LLM-as-Judge)
# =========================================================================

def judge_faithfulness(query, answer, context):
    """
    Is the answer grounded in the retrieved context?
    Uses GPT-4o-mini as a judge with a structured rubric.
    Returns: {"score": 1-5, "reason": "explanation"}

    TODO: Implement this in Session 1.
    """
    prompt = f"""You are an evaluation judge. Score whether the answer is grounded in the context.

Question: {query}
Context: {context}
Answer: {answer}

Score 5: every claim explicitly supported by context
Score 3: some claims not in context
Score 1: fabricated information

Return JSON only: {{"score": 1-5, "reason": "explanation"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def judge_correctness(query, answer, expected_answer):
    """
    Does the answer match the expected answer?
    Uses GPT-4o-mini as a judge.
    Returns: {"score": 1-5, "reason": "explanation"}

    TODO: Implement this in Session 1.
    """

    prompt = f"""You are an evaluation judge. Score whether the answer correctly addresses the question compared to the expected answer.

Question: {query}
Expected Answer: {expected_answer}
Generated Answer: {answer}

Score 5: answer fully matches expected answer
Score 3: partially correct
Score 1: completely wrong

Return JSON only: {{"score": 1-5, "reason": "explanation"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)



# =========================================================================
# EVAL RUNNER
# =========================================================================

def run_eval():
    """
    Run the full evaluation:
    1. Load golden dataset
    2. Run each query through the RAG pipeline
    3. Score retrieval (hit rate, MRR)
    4. Score generation (faithfulness, correctness)
    5. Print scorecard

    TODO: Implement this in Session 1.
    """
    dataset = load_golden_dataset()
    if not dataset:
        return

    results = []

    for item in dataset:
        print(f"Evaluating: {item['query'][:60]}...")
        result = ask(item['query'])
        hit = check_retrieval_hit(result['retrieved_chunks'], item['expected_source'])
        mrr = calculate_mrr(result['retrieved_chunks'], item['expected_source'])
        faith = judge_faithfulness(item['query'], result['answer'], result['context'])
        correct = judge_correctness(item['query'], result['answer'], item['expected_answer'])

        results.append({
            "id": item['id'],
            "query": item['query'],
            "expected_source": item['expected_source'],
            "answer": result['answer'],
            "trace_id": result['trace_id'],
            "hit": hit,
            "mrr": mrr,
            "faithfulness": faith['score'],
            "faithfulness_reason": faith['reason'],
            "correctness": correct['score'],
            "correctness_reason": correct['reason'],
        })

    hit_rate = sum(r['hit'] for r in results) / len(results) * 100
    avg_mrr = sum(r['mrr'] for r in results) / len(results)
    avg_faith = sum(r['faithfulness'] for r in results) / len(results)
    avg_correct = sum(r['correctness'] for r in results) / len(results)

    print("\n" + "="*50)
    print("EVALUATION SCORECARD")
    print("="*50)
    print(f"Queries evaluated:     {len(results)}")
    print(f"Hit Rate:              {hit_rate:.1f}%")
    print(f"Mean Reciprocal Rank:  {avg_mrr:.3f}")
    print(f"Avg Faithfulness:      {avg_faith:.2f} / 5")
    print(f"Avg Correctness:       {avg_correct:.2f} / 5")
    print("="*50)

    with open(os.path.join(SCRIPT_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to eval_results.json")




if __name__ == "__main__":
    run_eval()
    print("Eval harness skeleton loaded.")
    print("Functions to implement: check_retrieval_hit, calculate_mrr,")
    print("judge_faithfulness, judge_correctness, run_eval")
    print("\nWe'll build these together in Session 1.")
