import os
import json
from typing import List
from datetime import datetime
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

OPENROUTER_API_KEY = os.getenv("openrouter_api_key")


# ── RAGAS NEEDS LLM + EMBEDDINGS ─────────────────────────────────
def get_ragas_llm():
    llm = ChatOpenAI(
        model="google/gemma-3-27b-it:free",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=4096,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG POC"
        }
    )
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return LangchainEmbeddingsWrapper(embeddings)


# ── CORE EVALUATOR ────────────────────────────────────────────────
def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: List[str],       # list of retrieved chunk texts
    ground_truth: str = None   # optional: ideal answer for recall scoring
) -> dict:
    """
    Evaluate a single RAG response using RAGAS metrics.
    Returns a dict with all scores.
    """
    print(f"\n📊 Running RAGAS evaluation...")

    # Build dataset in RAGAS expected format
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],   # list of lists
    }

    # Ground truth is needed for context_recall
    if ground_truth:
        data["ground_truth"] = [ground_truth]
        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ]
    else:
        # Without ground truth, skip context_recall
        metrics = [
            context_precision,
            faithfulness,
            answer_relevancy
        ]

    dataset = Dataset.from_dict(data)

    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings()
    )

    # Convert to clean dict
    scores = {
        "question": question,
        "answer_preview": answer[:200],
        "timestamp": datetime.now().isoformat(),
        "context_precision": round(float(result["context_precision"]), 3),
        "faithfulness": round(float(result["faithfulness"]), 3),
        "answer_relevancy": round(float(result["answer_relevancy"]), 3),
    }

    if ground_truth:
        scores["context_recall"] = round(float(result["context_recall"]), 3)

    # Overall score (average of all metrics)
    metric_values = [v for k, v in scores.items()
                     if k not in ["question", "answer_preview", "timestamp"]]
    scores["overall_score"] = round(sum(metric_values) / len(metric_values), 3)

    # Print results
    print(f"\n{'='*50}")
    print(f"📊 RAGAS Scores for: '{question[:60]}...'")
    print(f"{'='*50}")
    print(f"  Context Precision : {scores['context_precision']} {'✅' if scores['context_precision'] > 0.7 else '⚠️'}")
    print(f"  Faithfulness      : {scores['faithfulness']} {'✅' if scores['faithfulness'] > 0.7 else '⚠️'}")
    print(f"  Answer Relevancy  : {scores['answer_relevancy']} {'✅' if scores['answer_relevancy'] > 0.7 else '⚠️'}")
    if ground_truth:
        print(f"  Context Recall    : {scores['context_recall']} {'✅' if scores['context_recall'] > 0.7 else '⚠️'}")
    print(f"  Overall Score     : {scores['overall_score']}")
    print(f"{'='*50}")

    return scores


# ── BATCH EVALUATOR ───────────────────────────────────────────────
def run_benchmark(test_cases: List[dict]) -> List[dict]:
    """
    Run RAGAS evaluation on a list of test cases.

    Each test case should be:
    {
        "question": "...",
        "answer": "...",
        "contexts": ["chunk1", "chunk2", ...],
        "ground_truth": "..." (optional)
    }
    """
    print(f"\n🚀 Running benchmark on {len(test_cases)} test cases...")
    all_scores = []

    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Evaluating...")
        scores = evaluate_rag_response(
            question=case["question"],
            answer=case["answer"],
            contexts=case["contexts"],
            ground_truth=case.get("ground_truth")
        )
        all_scores.append(scores)

    # Summary stats
    print(f"\n{'='*50}")
    print("📈 BENCHMARK SUMMARY")
    print(f"{'='*50}")
    for metric in ["context_precision", "faithfulness", "answer_relevancy", "overall_score"]:
        values = [s[metric] for s in all_scores if metric in s]
        if values:
            avg = round(sum(values) / len(values), 3)
            print(f"  Avg {metric:<22}: {avg}")

    return all_scores