"""
PDF parsing for NCERT Class 12 Physics Part 1.

Preserves: page numbers, chapter headings, section headings, formulas,
table content, and figure captions.

Why pdfplumber over pypdf:
- Better layout-aware text extraction (preserves column structure)
- Built-in table detection
- More accurate spacing/line break reconstruction

Why heading-aware chunking:
- Physics textbooks have clear chapter/section structure
- Chunking at semantic boundaries avoids splitting mid-concept
- Preserves chapter context for graph node creation
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError("pip install pdfplumber")


@dataclass
class TextBlock:
    """A single extracted text block from the PDF."""
    page_number: int
    chapter: str
    section: str
    subsection: str
    text: str
    block_type: str          # "paragraph" | "heading" | "formula" | "table" | "caption"
    raw_bbox: Optional[tuple] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Regex patterns for NCERT Physics structure
CHAPTER_PATTERN = re.compile(r"^chapter\s+(\d+)", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^(\d+\.\d+)\s+(.+)$")
SUBSECTION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+)\s+(.+)$")
FORMULA_PATTERNS = [
    re.compile(r"[=\+\-\*/]\s*[A-Za-z]\s*[=\+\-\*/]"),    # equations with operators
    re.compile(r"\b[A-Z][a-z]?\s*=\s*[A-Za-z0-9\+\-\*/\^\(\)]+"),  # variable = expr
    re.compile(r"∫|∑|∂|√|∞|α|β|γ|δ|ε|θ|λ|μ|π|σ|τ|φ|ω"),          # Greek/math symbols
]


def is_formula(text: str) -> bool:
    return any(p.search(text) for p in FORMULA_PATTERNS)


def is_heading(text: str, font_size: Optional[float] = None, font_name: str = "") -> bool:
    if font_size and font_size > 13:
        return True
    if "Bold" in font_name or "bold" in font_name:
        if len(text) < 120:
            return True
    if CHAPTER_PATTERN.match(text.strip()):
        return True
    if SECTION_PATTERN.match(text.strip()) or SUBSECTION_PATTERN.match(text.strip()):
        return True
    return False


def extract_blocks(pdf_path: str, max_pages: Optional[int] = None) -> list[TextBlock]:
    """
    Extract all text blocks from the PDF with structural metadata.
    """
    blocks = []
    current_chapter = "Unknown Chapter"
    current_section = ""
    current_subsection = ""

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]

        for page in pages:
            page_num = page.page_number
            text = page.extract_text(x_tolerance=3, y_tolerance=3)

            if not text:
                continue

            # Try to extract tables separately
            tables = page.extract_tables()
            table_texts = set()
            for table in tables:
                table_text = "\n".join(
                    " | ".join(str(cell or "") for cell in row)
                    for row in table if row
                )
                if table_text.strip():
                    table_texts.add(table_text.strip()[:200])  # first 200 chars for dedup
                    blocks.append(TextBlock(
                        page_number=page_num,
                        chapter=current_chapter,
                        section=current_section,
                        subsection=current_subsection,
                        text=table_text,
                        block_type="table",
                    ))

            # Process line by line
            for line in text.split("\n"):
                line = line.strip()
                if not line or len(line) < 3:
                    continue

                # Detect headings
                if CHAPTER_PATTERN.match(line):
                    current_chapter = line
                    current_section = ""
                    current_subsection = ""
                    blocks.append(TextBlock(
                        page_number=page_num,
                        chapter=current_chapter,
                        section="",
                        subsection="",
                        text=line,
                        block_type="heading",
                    ))
                    continue

                sec_match = SUBSECTION_PATTERN.match(line)
                if sec_match:
                    current_subsection = line
                    blocks.append(TextBlock(
                        page_number=page_num,
                        chapter=current_chapter,
                        section=current_section,
                        subsection=current_subsection,
                        text=line,
                        block_type="heading",
                    ))
                    continue

                sec_match = SECTION_PATTERN.match(line)
                if sec_match:
                    current_section = line
                    current_subsection = ""
                    blocks.append(TextBlock(
                        page_number=page_num,
                        chapter=current_chapter,
                        section=current_section,
                        subsection="",
                        text=line,
                        block_type="heading",
                    ))
                    continue

                # Detect formulas
                block_type = "formula" if is_formula(line) else "paragraph"

                # Skip if this text already captured in a table
                if any(line[:50] in t for t in table_texts):
                    continue

                blocks.append(TextBlock(
                    page_number=page_num,
                    chapter=current_chapter,
                    section=current_section,
                    subsection=current_subsection,
                    text=line,
                    block_type=block_type,
                ))

    print(f"Extracted {len(blocks)} blocks from {pdf_path}")
    return blocks


def save_blocks(blocks: list[TextBlock], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([b.to_dict() for b in blocks], f, indent=2, ensure_ascii=False)
    print(f"Saved {len(blocks)} blocks to {output_path}")


def load_blocks(path: str) -> list[TextBlock]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [TextBlock(**d) for d in data]
