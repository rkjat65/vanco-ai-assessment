"""
RAG evaluation: retrieval quality, answer faithfulness, and latency.

Evaluation dimensions:
1. Retrieval recall@k    — are relevant chunks retrieved?
2. Answer faithfulness   — does the answer match the context?
3. Answer relevance      — does the answer address the question?
4. Latency              — end-to-end response time

Uses a hand-curated question set covering:
- Factual questions (specific values, definitions)
- Conceptual questions (explanations, principles)
- Formula-based questions (derive, state, calculate)
- Comparison questions (compare A vs B)
- Cross-chapter questions (information from multiple sections)
"""

import sys
import time
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# Hand-curated evaluation set for NCERT Physics Part 1
EVAL_QUESTIONS = [
    {
        "question": "State Gauss's Law in electrostatics.",
        "expected_keywords": ["flux", "enclosed", "charge", "epsilon", "surface"],
        "expected_chapter": "Electric Charges",
        "type": "factual",
    },
    {
        "question": "What is the electric field inside a hollow conductor?",
        "expected_keywords": ["zero", "conductor", "inside", "shielding"],
        "expected_chapter": "Electric Charges",
        "type": "factual",
    },
    {
        "question": "Derive the expression for capacitance of a parallel plate capacitor.",
        "expected_keywords": ["capacitance", "area", "distance", "permittivity", "epsilon"],
        "expected_chapter": "Electrostatic Potential",
        "type": "formula",
    },
    {
        "question": "State Ohm's Law and its limitations.",
        "expected_keywords": ["voltage", "current", "resistance", "temperature", "linear"],
        "expected_chapter": "Current Electricity",
        "type": "conceptual",
    },
    {
        "question": "What is the difference between emf and terminal voltage?",
        "expected_keywords": ["emf", "internal resistance", "terminal", "load"],
        "expected_chapter": "Current Electricity",
        "type": "comparison",
    },
    {
        "question": "State Faraday's laws of electromagnetic induction.",
        "expected_keywords": ["flux", "emf", "change", "rate", "coil"],
        "expected_chapter": "Electromagnetic Induction",
        "type": "factual",
    },
    {
        "question": "What is Lenz's Law?",
        "expected_keywords": ["oppose", "change", "flux", "induced", "current"],
        "expected_chapter": "Electromagnetic Induction",
        "type": "factual",
    },
    {
        "question": "Explain the photoelectric effect.",
        "expected_keywords": ["photon", "electron", "threshold", "frequency", "work function"],
        "expected_chapter": "Dual Nature",
        "type": "conceptual",
    },
    {
        "question": "What is de Broglie wavelength?",
        "expected_keywords": ["wavelength", "momentum", "wave", "particle", "h"],
        "expected_chapter": "Dual Nature",
        "type": "formula",
    },
    {
        "question": "What happens to capacitance when a dielectric is inserted?",
        "expected_keywords": ["dielectric", "increases", "constant", "k", "polarization"],
        "expected_chapter": "Electrostatic Potential",
        "type": "conceptual",
    },
]


def recall_at_k(chunks: list[dict], expected_keywords: list[str], k: int = 5) -> float:
    """
    Compute keyword recall: what fraction of expected keywords appear
    in the top-k retrieved chunks?
    """
    top_k_text = " ".join(c.get("text", "") for c in chunks[:k]).lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in top_k_text)
    return found / len(expected_keywords) if expected_keywords else 0.0


def answer_faithfulness(answer: str, chunks: list[dict]) -> float:
    """
    Naive faithfulness: what fraction of the answer sentences contain
    at least one phrase from the source chunks?
    (Production version would use an NLI model or LLM-as-judge)
    """
    source_text = " ".join(c.get("text", "") for c in chunks).lower()
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]

    if not sentences:
        return 0.0

    # A sentence is "grounded" if ≥ 3-gram overlap with source
    grounded = 0
    for sentence in sentences:
        words = sentence.lower().split()
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        if any(tg in source_text for tg in trigrams):
            grounded += 1

    return grounded / len(sentences)


def run_evaluation(
    index_dir: str = "indexes/",
    output_path: str = "evaluation/results.json",
    questions: Optional[list[dict]] = None,
):
    from retrieval.hybrid_retriever import HybridRetriever
    from generation.answer_generator import AnswerGenerator
    from dotenv import load_dotenv
    load_dotenv()

    questions = questions or EVAL_QUESTIONS

    print("Loading pipeline...")
    retriever = HybridRetriever(index_dir=index_dir)
    retriever.load()
    generator = AnswerGenerator()

    results = []

    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"\n[{i}/{len(questions)}] {q}")

        t0 = time.time()
        retrieval_result = retriever.retrieve(q, final_top_n=6)
        t_retrieve = time.time() - t0

        t1 = time.time()
        answer_data = generator.generate_with_retrieval_trace(q, retrieval_result)
        t_generate = time.time() - t1

        chunks = retrieval_result.get("chunks", [])
        recall = recall_at_k(chunks, item.get("expected_keywords", []))
        faith = answer_faithfulness(answer_data["answer"], chunks)

        result = {
            "question": q,
            "type": item.get("type"),
            "answer": answer_data["answer"],
            "citations": answer_data.get("citations", []),
            "recall_at_5": round(recall, 3),
            "faithfulness": round(faith, 3),
            "latency_retrieve_ms": round(t_retrieve * 1000, 1),
            "latency_generate_ms": round(t_generate * 1000, 1),
            "latency_total_ms": round((t_retrieve + t_generate) * 1000, 1),
            "chunks_used": len(chunks),
            "input_tokens": answer_data.get("input_tokens", 0),
        }
        results.append(result)

        print(f"  Recall@5: {recall:.2f} | Faithfulness: {faith:.2f} | "
              f"Total latency: {result['latency_total_ms']:.0f}ms")

    # Aggregate stats
    avg_recall = sum(r["recall_at_5"] for r in results) / len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_latency = sum(r["latency_total_ms"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"  Questions evaluated: {len(results)}")
    print(f"  Avg Recall@5:        {avg_recall:.3f}")
    print(f"  Avg Faithfulness:    {avg_faith:.3f}")
    print(f"  Avg Total Latency:   {avg_latency:.0f}ms")
    print("="*60)

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_recall_at_5": avg_recall,
                "avg_faithfulness": avg_faith,
                "avg_latency_ms": avg_latency,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="indexes/")
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()
    run_evaluation(args.index_dir, args.output)
