"""Semantic search over the Pathfinder astronomy corpus (Iyer et al. 2024,
arXiv:2408.01556). Reads pre-computed text-embedding-3-small vectors from
either the mounted HF bucket (/data/data) or a local cache, builds a FAISS
index over the embed column, and returns ranked papers in foto's standard
paper-dict format."""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from datasets import load_dataset, load_from_disk


EMBEDDING_MODEL = "text-embedding-3-small"
HF_DATASET_NAME = "kiyer/pathfinder_arxiv_data"

BUCKET_PATH = Path("/data/data")
LOCAL_CACHE = Path.home() / ".cache" / "foto" / "pathfinder_data"


@st.cache_resource(show_spinner="Loading Pathfinder corpus...")
def load_pathfinder_corpus():
    if BUCKET_PATH.exists():
        ds = load_dataset(
            "parquet",
            data_files=str(BUCKET_PATH / "*.parquet"),
            split="train",
        )
    elif LOCAL_CACHE.exists():
        ds = load_from_disk(str(LOCAL_CACHE))
    else:
        LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(HF_DATASET_NAME, split="train")
        ds.save_to_disk(str(LOCAL_CACHE))

    if not ds.is_index_initialized("embed"):
        ds.add_faiss_index(column="embed")

    return ds


def make_embedder(openai_key: str):
    from openai import OpenAI

    if not openai_key:
        raise RuntimeError(
            "Pathfinder uses text-embedding-3-small from OpenAI. Provide an "
            "OpenAI API key in the sidebar (get one at platform.openai.com)."
        )
    client = OpenAI(api_key=openai_key)

    def embed(text: str) -> np.ndarray:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return np.array(resp.data[0].embedding, dtype=np.float32)

    return embed


def _row_to_paper(row: dict, similarity: float) -> dict:
    arxiv_id = row.get("arxiv_id") or ""

    year = None
    d = row.get("date")
    if d is not None:
        try:
            year = d.year
        except AttributeError:
            s = str(d)
            year = int(s[:4]) if s[:4].isdigit() else None

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
    def __init__(self, openai_key: str):
        self.dataset = load_pathfinder_corpus()
        self.embed = make_embedder(openai_key)

    def search(self, query: str, limit: int = 50) -> list[dict]:
        query_vec = self.embed(query)
        tmp = self.dataset.search("embed", query_vec, k=limit)

        results = []
        for idx, dist in zip(tmp.indices, tmp.scores):
            row = self.dataset[int(idx)]
            if not row.get("arxiv_id"):
                continue
            similarity = 1.0 / (1.0 + float(dist))
            results.append(_row_to_paper(row, similarity))

        return results
