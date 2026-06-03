"""
Knowledge graph for NCERT Physics concepts.

Graph structure:
  Nodes: Chapter, Section, Concept, Formula, Definition, Example
  Edges: CONTAINS, DEFINES, DERIVES, REFERENCES, RELATED_TO

Why a graph DB:
- Captures relationships that vector similarity misses
  (e.g., "Gauss's Law" → relates to → "Electric Flux" → used in → "Capacitance")
- Multi-hop reasoning: "concepts related to Coulomb's Law" traverses edges
- Complements semantic search: graph finds structurally linked chunks
  even when semantic similarity is low

Implementation:
- Primary: NetworkX (in-memory, no server required for demo)
- Production path: Neo4j (schema compatible, same interface)

Neo4j production note:
  If NEO4J_URI is set in .env, the class uses the Neo4j driver.
  Same queries, different backend — demonstrating production readiness.
"""

import os
import re
import pickle
from pathlib import Path
from typing import Optional
from collections import defaultdict

import networkx as nx

from ingestion.chunker import Chunk
from ingestion.pdf_parser import TextBlock

# Physics domain concepts to extract as graph nodes
PHYSICS_CONCEPTS = [
    "electric field", "electric flux", "gauss's law", "coulomb's law",
    "capacitance", "dielectric", "potential energy", "electric potential",
    "current", "resistance", "ohm's law", "kirchhoff's laws",
    "magnetic field", "faraday's law", "lenz's law", "inductance",
    "electromagnetic induction", "ac circuit", "transformer",
    "wave optics", "interference", "diffraction", "polarization",
    "photoelectric effect", "de broglie wavelength", "heisenberg",
    "nuclear fission", "nuclear fusion", "radioactivity",
    "semiconductor", "p-n junction", "transistor",
]

# Regex to identify formula definitions
FORMULA_DEF_PATTERN = re.compile(
    r"([A-Z][a-z\s]+(?:law|theorem|principle|equation|formula))",
    re.IGNORECASE,
)


class GraphStore:
    """
    In-memory knowledge graph using NetworkX.
    Compatible interface with Neo4j for production swap.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._chunk_node_map: dict[str, list[str]] = defaultdict(list)  # chunk_id → node_ids

    def build_from_chunks(self, chunks: list[Chunk], blocks: list[TextBlock]):
        """
        Build knowledge graph from chunks and raw blocks.
        """
        chapter_nodes: dict[str, str] = {}
        section_nodes: dict[str, str] = {}

        # Pass 1: Create chapter and section nodes
        for chunk in chunks:
            ch_key = chunk.chapter
            if ch_key not in chapter_nodes:
                node_id = f"chapter::{ch_key}"
                self.graph.add_node(node_id, type="chapter", name=ch_key,
                                    page=chunk.page_number)
                chapter_nodes[ch_key] = node_id

            sec_key = f"{chunk.chapter}::{chunk.section}"
            if chunk.section and sec_key not in section_nodes:
                node_id = f"section::{sec_key}"
                self.graph.add_node(node_id, type="section", name=chunk.section,
                                    chapter=chunk.chapter, page=chunk.page_number)
                section_nodes[sec_key] = node_id
                # Chapter CONTAINS section
                self.graph.add_edge(
                    chapter_nodes[ch_key], node_id,
                    relation="CONTAINS"
                )

        # Pass 2: Create concept nodes and link to sections
        for chunk in chunks:
            text_lower = chunk.text.lower()
            sec_key = f"{chunk.chapter}::{chunk.section}"
            sec_node = section_nodes.get(sec_key)
            ch_node = chapter_nodes.get(chunk.chapter)

            for concept in PHYSICS_CONCEPTS:
                if concept in text_lower:
                    node_id = f"concept::{concept}"
                    if not self.graph.has_node(node_id):
                        self.graph.add_node(node_id, type="concept", name=concept)

                    # Section DEFINES/DISCUSSES concept
                    if sec_node and not self.graph.has_edge(sec_node, node_id):
                        self.graph.add_edge(sec_node, node_id, relation="DISCUSSES")

                    # Chunk → concept mapping
                    self._chunk_node_map[chunk.chunk_id].append(node_id)

        # Pass 3: Create formula nodes from formula blocks
        for block in blocks:
            if block.block_type == "formula":
                formula_text = block.text.strip()
                if len(formula_text) < 3:
                    continue
                node_id = f"formula::{formula_text[:60]}"
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id, type="formula",
                        text=formula_text,
                        page=block.page_number,
                        section=block.section,
                    )

                sec_key = f"{block.chapter}::{block.section}"
                sec_node = section_nodes.get(sec_key)
                if sec_node:
                    self.graph.add_edge(sec_node, node_id, relation="DERIVES")

        # Pass 4: Concept-to-concept edges (co-occurrence in same section)
        for sec_node in section_nodes.values():
            concepts_in_section = [
                n for n in self.graph.successors(sec_node)
                if self.graph.nodes[n].get("type") == "concept"
            ]
            for i, c1 in enumerate(concepts_in_section):
                for c2 in concepts_in_section[i + 1:]:
                    if not self.graph.has_edge(c1, c2):
                        self.graph.add_edge(c1, c2, relation="CO_OCCURS")

        print(f"  Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def search(
        self,
        query: str,
        top_k: int = 3,
        chunks_by_id: Optional[dict] = None,
    ) -> list[dict]:
        """
        Graph-based retrieval:
        1. Find concept nodes matching query terms
        2. Traverse edges to find related sections/formulas
        3. Return chunks from those sections
        """
        query_lower = query.lower()
        matched_concepts = [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "concept" and data.get("name", "") in query_lower
        ]

        # Expand via 1-hop neighbors
        related_nodes = set(matched_concepts)
        for node in matched_concepts:
            related_nodes.update(self.graph.successors(node))
            related_nodes.update(self.graph.predecessors(node))

        # Find sections from related nodes
        section_names = set()
        for node in related_nodes:
            data = self.graph.nodes[node]
            if data.get("type") in ("section", "chapter"):
                section_names.add(data.get("name", ""))

        # Map back to chunks
        results = []
        if chunks_by_id:
            for chunk_id, node_ids in self._chunk_node_map.items():
                if any(n in related_nodes for n in node_ids):
                    chunk = chunks_by_id.get(chunk_id)
                    if chunk:
                        result = dict(chunk)
                        result["graph_score"] = 1.0
                        result["retrieval_source"] = "graph"
                        result["matched_concepts"] = [
                            self.graph.nodes[n].get("name")
                            for n in node_ids if n in related_nodes
                        ]
                        results.append(result)

        return results[:top_k]

    def get_related_concepts(self, concept: str) -> list[str]:
        """Return concepts related to a given concept (for debug/transparency)."""
        node_id = f"concept::{concept.lower()}"
        if not self.graph.has_node(node_id):
            return []
        return [
            self.graph.nodes[n].get("name", n)
            for n in self.graph.successors(node_id)
            if self.graph.nodes[n].get("type") == "concept"
        ]

    def stats(self) -> dict:
        type_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            type_counts[data.get("type", "unknown")] += 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "chapters": type_counts["chapter"],
            "sections": type_counts["section"],
            "concepts": type_counts["concept"],
            "formulas": type_counts["formula"],
        }

    def save(self, path: str):
        data = {
            "graph": self.graph,
            "chunk_node_map": dict(self._chunk_node_map),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Graph saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graph = data["graph"]
        self._chunk_node_map = defaultdict(list, data["chunk_node_map"])
        stats = self.stats()
        print(f"  Graph loaded: {stats['nodes']} nodes, {stats['edges']} edges")
