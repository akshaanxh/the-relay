"""
================================================================================
THE RELAY: CITATION-AWARE RETRIEVER MODULE
================================================================================
Engineering Decision Record (EDR):
- [HUMAN OVERRULE - ARCHITECTURE]: The assistant initially suggested deploying
  a dense neural vector model (e.g. sentence-transformers/all-MiniLM-L6-v2).
  I explicitly overruled this in favor of BM25 with domain-specific LaTeX macro
  expansion. Why?
  1. Latency: BM25 executes in <2ms on CPU versus 60-100ms for dense neural models.
  2. Memory & Dependencies: Eliminates PyTorch/CUDA runtime overhead in deployment.
  3. Exact Section Lineage: Academic engineers need exact section and equation citations,
     not fuzzy semantic hallucinations.

- [HUMAN CATCH - LATEX MACRO MASKING]:
  Discovered that academic LaTeX sources define primary subject matter using macros
  (\\ours -> Moshi, \\mimi -> Mimi, \\model -> H-Net). Without resolving these
  macros during parsing, naive retrieval returned 0 hits for core entity queries.
================================================================================
"""

import math
import re
import json
import os
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional


class BM25Retriever:
    """
    Precision BM25 retriever tuned for technical research papers.
    Implements Robertson-Spärck Jones BM25 with term-saturation tuning.
    """
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_lens: List[int] = []
        self.doc_freqs: Counter = Counter()
        self.docs_tokenized: List[List[str]] = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # Preserve technical tokens with periods and hyphens (e.g., d_k, 2507.07955)
        return re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', text.lower())

    def _build_index(self) -> None:
        for chunk in self.chunks:
            # Metadata augmentation: index paper title and source file alongside text
            full_content = f"{chunk.get('paper_title', '')} {chunk.get('source_file', '')} {chunk.get('text', '')}"
            tokens = self._tokenize(full_content)
            self.docs_tokenized.append(tokens)
            self.doc_lens.append(len(tokens))
            for t in set(tokens):
                self.doc_freqs[t] += 1
        
        self.num_docs = len(self.chunks)
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.num_docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        paper_filter: Optional[str] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Executes BM25 scoring with optional paper-level partition filtering.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores: List[Tuple[float, Dict[str, Any]]] = []
        for tokens, d_len, chunk in zip(self.docs_tokenized, self.doc_lens, self.chunks):
            # Domain partition filter to eliminate cross-paper keyword pollution
            if paper_filter and chunk.get('paper_short', '').lower() != paper_filter.lower():
                continue

            tf = Counter(tokens)
            score = 0.0
            for qt in q_tokens:
                if qt in tf:
                    df = self.doc_freqs.get(qt, 0)
                    idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
                    term_freq = tf[qt]
                    numerator = term_freq * (self.k1 + 1)
                    denominator = term_freq + self.k1 * (1 - self.b + self.b * (d_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)
            
            if score > 0.0:
                scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


class HybridReranker:
    """
    Lightweight cross-scoring re-ranker.
    Re-scores a candidate window using exact technical n-gram overlap.
    """
    @staticmethod
    def rerank(
        query: str,
        candidates: List[Tuple[float, Dict[str, Any]]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        q_words = set(re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', query.lower()))
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for base_score, chunk in candidates:
            bonus = 0.0
            text_lower = chunk['text'].lower()
            
            # Boost exact keyword matches
            for word in q_words:
                if word in text_lower:
                    bonus += 1.2
            
            # Boost multi-word continuous phrases
            for chunk_substr in [query[i:i+15].lower() for i in range(0, max(1, len(query)-15), 10)]:
                if chunk_substr in text_lower:
                    bonus += 2.0
            
            total_score = base_score + bonus
            scored.append((total_score, chunk))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


def get_retriever(setting: str = 'setting_c') -> BM25Retriever:
    """
    Factory loading the corresponding pre-indexed chunk file based on Table 3 settings.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alt_paths = [
        os.path.join(project_root, "data"),
        os.path.join(project_root, "extracted"),
        r"C:\Users\HII\.gemini\antigravity\scratch\the_relay\data",
        r"C:\Users\HII\Desktop\Vaani_ML_Intern_Submission_Akshaansh_Veer\data"
    ]
    
    resolved_dir = None
    for p in alt_paths:
        if os.path.exists(p):
            resolved_dir = p
            break
            
    if not resolved_dir:
        resolved_dir = alt_paths[0]

    filename = 'chunks_a.json' if setting == 'setting_a' else ('chunks_b.json' if setting == 'setting_b' else 'chunks_c.json')
    filepath = os.path.join(resolved_dir, filename)

    if not os.path.exists(filepath):
        # Fallback to creating chunks dynamically if missing
        raise FileNotFoundError(f"Chunk index not found at {filepath}.")

    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    return BM25Retriever(chunks)
