"""
Grounded answer generation using Claude (Anthropic API).

Critical design: The system prompt enforces strict grounding.
The model must ONLY answer based on provided context chunks.
If the answer isn't in the context, it says so explicitly.

Why Claude over GPT-4 or open-source models:
- Superior at following strict grounding instructions
- Better citation extraction and reference formatting
- Handles physics formulas in LaTeX/text form well
- Context window (200K tokens) handles large retrieval sets
- Trade-off: costs money; for cost-free option, use llama.cpp + Mistral
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

try:
    import anthropic
except ImportError:
    raise ImportError("pip install anthropic")

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a physics tutor assistant grounded EXCLUSIVELY in the provided NCERT Class 12 Physics textbook excerpts.

STRICT RULES:
1. Answer ONLY from the provided context. Do not use any prior knowledge.
2. If the answer is NOT in the provided context, respond with:
   "This information is not available in the provided NCERT Physics document."
3. Always cite the source: chapter name and page number(s) from the context.
4. For formulas, reproduce them exactly as they appear in the context.
5. Be precise and educational. Structure answers clearly.
6. For comparison questions, use a structured format with clear distinctions.
7. Indicate confidence: if only partially covered in context, say so.

FORMAT:
- Lead with the direct answer
- Support with relevant quotes or formulas from context
- End with [Source: Chapter X, page Y] citation(s)
"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered context for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        citation = chunk.get("citation", f"p.{chunk.get('page_number', '?')}")
        source_tag = chunk.get("retrieval_source", "unknown")
        lines.append(
            f"[Context {i}] [{citation}] (retrieved via: {source_tag})\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(lines)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_str = format_context(chunks)
    return (
        f"CONTEXT FROM NCERT PHYSICS TEXTBOOK:\n"
        f"{'─'*60}\n"
        f"{context_str}\n"
        f"{'─'*60}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer based strictly on the above context:"
    )


class AnswerGenerator:
    def __init__(self, model: str = MODEL):
        # Load from .env, overriding empty Claude Code env vars
        from dotenv import dotenv_values
        env_vals = dotenv_values(Path(__file__).parent.parent / ".env")
        api_key = env_vals.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key or api_key == "your_key_here":
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        # Clear conflicting env vars that Claude Code sets to empty strings
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        question: str,
        chunks: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.1,  # low temperature for factual answers
    ) -> dict:
        """
        Generate a grounded answer from retrieved chunks.

        Returns:
            {
                "answer": str,
                "citations": list[str],
                "input_tokens": int,
                "output_tokens": int,
            }
        """
        if not chunks:
            return {
                "answer": "No relevant content was retrieved from the document for this question.",
                "citations": [],
                "input_tokens": 0,
                "output_tokens": 0,
            }

        prompt = build_prompt(question, chunks)

        response = self.client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        answer_text = response.content[0].text

        # Extract citations from chunks used
        citations = list({
            chunk.get("citation", f"p.{chunk.get('page_number', '?')}")
            for chunk in chunks
        })

        return {
            "answer": answer_text,
            "citations": citations,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "chunks_used": len(chunks),
        }

    def generate_with_retrieval_trace(
        self,
        question: str,
        retrieval_result: dict,
    ) -> dict:
        """
        Full pipeline: retrieval result → grounded answer with debug trace.
        """
        chunks = retrieval_result.get("chunks", [])
        answer_data = self.generate(question, chunks)

        answer_data["retrieval_debug"] = retrieval_result.get("debug", {})
        answer_data["sources"] = [
            {
                "chunk_id": c.get("chunk_id"),
                "citation": c.get("citation"),
                "retrieval_source": c.get("retrieval_source"),
                "score": c.get("rerank_score") or c.get("rrf_score"),
                "text_preview": c.get("text", "")[:150] + "...",
            }
            for c in chunks
        ]

        return answer_data
