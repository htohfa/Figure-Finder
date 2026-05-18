"""Semantic search over the Pathfinder astronomy corpus (Iyer et al. 2024,
arXiv:2408.01556).

The corpus ships with pre-computed text-embedding-3-small vectors for each
paper. Queries are embedded with the same OpenAI model and matched via FAISS
on the embedding column. Output format matches the rest of foto's search
layer so it slots in interchangeably with keyword search.
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from datasets import load_from_disk, load_dataset


DATASET_NAME = "kiyer/pathfinder_arxiv_data"
EMBEDDING_MODEL = "text-embedding-3-small"

# First-run download lands here; subsequent runs load_from_disk straight from cache
DATA_DIR = Path.home() / ".cache" / "foto" / "pathfinder_data"


@st.cache_resource(show_spinner="Loading Pathfinder corpus (~5 GB on first run)...")
def load_pathfinder_corpus():
    """Returns the dataset with a FAISS index attached to the embed column.
    Downloads from HF on first call, reuses local cache afterward."""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(DATASET_NAME, split="train")
        ds.save_to_disk(str(DATA_DIR))
    else:
        ds = load_from_disk(str(DATA_DIR))

    if not ds.is_index_initialized("embed"):
        ds.add_faiss_index(column="embed")

    return ds


def make_embedder(openai_key: str):
    """Returns a function that embeds text into a 1536-dim vector with
    text-embedding-3-small. Key is user-supplied so this is not cached."""
    from openai import OpenAI

    if not openai_key:
        raise RuntimeError(
            "Pathfinder uses text-embedding-3-small from OpenAI. Set an "
            "OpenAI API key in the sidebar (get one at platform.openai.com)."
        )
    client = OpenAI(api_key=openai_key)

    def embed(text: str) -> np.ndarray:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return np.array(resp.data[0].embedding, dtype=np.float32)

    return embed


def _row_to_paper(row: dict, similarity: float) -> dict:
    """Map a Pathfinder dataset row into foto's paper dict shape."""
    arxiv_id = row.get("arxiv_id") or ""

    year = None
    d = row.get("date")
    if d is not None:
        try:
            year = d.year
        except AttributeError:
            s = str(d)
            year = int(s[:4]) if s[:4].isdigit() else None

    # Pathfinder stores authors as a list of strings; foto wants [{"name": ...}, ...]
    raw_authors = row.get("authors") or []
    if raw_authors and isinstance(raw_authors[0], str):
        authors = [{"name": a} for a in raw_authors]
    else:
        authors = raw_authors

    return {
        "paperId": f"arxiv_{arxiv_id}" if arxiv_id else f"ads_{row.get('ads_id', '')}",
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "year": year,
        "authors": authors,
        "externalIds": {"ArXiv": arxiv_id} if arxiv_id else {},
        "openAccessPdf": {"url": f"https://arxiv.org/pdf/{arxiv_id}"} if arxiv_id else {},
        "citationCount": row.get("cites", 0) or 0,
        "_source": "pathfinder",
        "_pathfinder_score": similarity,
    }


class PathfinderSearcher:
    """Semantic retrieval over the Pathfinder corpus.
    Output format matches PaperSearcher.search_s2 so downstream code is unchanged."""

    def __init__(self, openai_key: str):
        self.dataset = load_pathfinder_corpus()
        self.embed = make_embedder(openai_key)

    def search(self, query: str, limit: int = 50) -> list[dict]:
        query_vec = self.embed(query)

        tmp = self.dataset.search("embed", query_vec, k=limit)

        results = []
        for idx, dist in zip(tmp.indices, tmp.scores):
            row = self.dataset[int(idx)]
            # Skip papers with no arxiv_id — the download step needs it
            if not row.get("arxiv_id"):
                continue
            # Convert FAISS distance to similarity, matching Pathfinder's convention
            similarity = 1.0 / (1.0 + float(dist))
            results.append(_row_to_paper(row, similarity))

        return results
