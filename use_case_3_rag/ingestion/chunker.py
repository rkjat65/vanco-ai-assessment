"""
Heading-aware chunking strategy for NCERT Physics PDF.

Chunking strategy:
- Primary: section-aware (chunk at section/subsection boundaries)
- Fallback: token-based sliding window with overlap
- Special: formulas are never split across chunks
- Tables are kept as single chunks regardless of size

Why heading-aware over naive character-split:
- Physics explanations build on context within a section
- Splitting mid-derivation produces useless fragments for retrieval
- Keeps citation references (page + section) meaningful
- Enables graph node creation at section granularity

Trade-offs:
- Sections vary widely in length → uneven chunk sizes
- Very long sections (e.g., Gauss's Law explanation) may need sub-splitting
- Short sections (just a formula + 2 lines) may not embed well
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from .pdf_parser import TextBlock


@dataclass
class Chunk:
    """A single chunk ready for embedding and retrieval."""
    chunk_id: str
    text: str
    page_number: int
    page_end: int
    chapter: str
    section: str
    subsection: str
    block_types: list[str]  # types of blocks merged into this chunk
    token_count: int        # approximate word count

    def citation(self) -> str:
        """Human-readable citation string."""
        parts = [self.chapter]
        if self.section:
            parts.append(self.section)
        if self.subsection:
            parts.append(self.subsection)
        page_str = (
            f"p.{self.page_number}"
            if self.page_number == self.page_end
            else f"pp.{self.page_number}–{self.page_end}"
        )
        return f"{' > '.join(parts)} ({page_str})"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_number": self.page_number,
            "page_end": self.page_end,
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "block_types": self.block_types,
            "token_count": self.token_count,
            "citation": self.citation(),
        }


def estimate_tokens(text: str) -> int:
    """Rough word count as token proxy (actual BPE tokens ≈ 1.3× words)."""
    return len(text.split())


def section_aware_chunks(
    blocks: list[TextBlock],
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """
    Merge consecutive blocks into chunks bounded by section headings.

    Algorithm:
    1. Walk through blocks in order
    2. At a section/chapter heading, flush the current buffer as a chunk
    3. Continue accumulating until max_tokens reached, then split
    4. Never split in the middle of a formula block
    5. Apply overlap by carrying the last `overlap_tokens` words into next chunk
    """
    chunks: list[Chunk] = []
    buffer: list[TextBlock] = []
    buffer_tokens = 0
    chunk_counter = 0

    def flush_buffer(buf: list[TextBlock]) -> Optional[Chunk]:
        nonlocal chunk_counter
        if not buf:
            return None

        merged_text = " ".join(b.text for b in buf)
        chunk = Chunk(
            chunk_id=f"chunk_{chunk_counter:05d}",
            text=merged_text,
            page_number=buf[0].page_number,
            page_end=buf[-1].page_number,
            chapter=buf[0].chapter,
            section=buf[0].section,
            subsection=buf[0].subsection,
            block_types=list({b.block_type for b in buf}),
            token_count=estimate_tokens(merged_text),
        )
        chunk_counter += 1
        return chunk

    def carry_overlap(buf: list[TextBlock], n_words: int) -> list[TextBlock]:
        """Keep last n_words from buffer as context overlap for next chunk."""
        all_text = " ".join(b.text for b in buf)
        words = all_text.split()
        if len(words) <= n_words:
            return buf
        overlap_text = " ".join(words[-n_words:])
        # Create a synthetic overlap block
        return [TextBlock(
            page_number=buf[-1].page_number,
            chapter=buf[-1].chapter,
            section=buf[-1].section,
            subsection=buf[-1].subsection,
            text=overlap_text,
            block_type="paragraph",
        )]

    for block in blocks:
        is_section_boundary = (
            block.block_type == "heading" and
            (block.chapter != (buffer[0].chapter if buffer else None) or
             block.section != (buffer[0].section if buffer else None))
        )

        # Flush at section boundary
        if is_section_boundary and buffer:
            chunk = flush_buffer(buffer)
            if chunk:
                chunks.append(chunk)
            buffer = carry_overlap(buffer, overlap_tokens)
            buffer_tokens = estimate_tokens(" ".join(b.text for b in buffer))

        buffer.append(block)
        buffer_tokens += estimate_tokens(block.text)

        # Flush when exceeding max tokens (but not in the middle of a formula)
        if buffer_tokens >= max_tokens and block.block_type != "formula":
            chunk = flush_buffer(buffer)
            if chunk:
                chunks.append(chunk)
            buffer = carry_overlap(buffer, overlap_tokens)
            buffer_tokens = estimate_tokens(" ".join(b.text for b in buffer))

    # Flush remaining buffer
    if buffer:
        chunk = flush_buffer(buffer)
        if chunk:
            chunks.append(chunk)

    print(f"Created {len(chunks)} chunks (max_tokens={max_tokens}, overlap={overlap_tokens})")
    return chunks


def create_formula_chunks(blocks: list[TextBlock]) -> list[Chunk]:
    """
    Create dedicated chunks for important formula blocks.

    Formulas often need direct retrieval (e.g., "what is Gauss's law formula?")
    without interference from surrounding explanatory text.
    """
    formula_chunks = []
    for i, block in enumerate(blocks):
        if block.block_type == "formula":
            # Include surrounding context (1 block before/after for meaning)
            context_before = blocks[i - 1].text if i > 0 else ""
            context_after = blocks[i + 1].text if i < len(blocks) - 1 else ""
            text = f"{context_before} {block.text} {context_after}".strip()

            formula_chunks.append(Chunk(
                chunk_id=f"formula_{i:05d}",
                text=text,
                page_number=block.page_number,
                page_end=block.page_number,
                chapter=block.chapter,
                section=block.section,
                subsection=block.subsection,
                block_types=["formula"],
                token_count=estimate_tokens(text),
            ))

    print(f"Created {len(formula_chunks)} formula-specific chunks")
    return formula_chunks


def all_chunks(blocks: list[TextBlock]) -> list[Chunk]:
    """
    Combine section-aware chunks and formula chunks.
    Formula chunks are appended with unique IDs for dedicated retrieval.
    """
    main = section_aware_chunks(blocks)
    formulas = create_formula_chunks(blocks)
    combined = main + formulas
    print(f"Total chunks: {len(combined)} ({len(main)} main + {len(formulas)} formula)")
    return combined
