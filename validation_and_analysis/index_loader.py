"""Load a FOTO caption index: vectors (FAISS or npy) plus row-aligned metadata,
joining full captions/titles from astro_captions.parquet when missing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIG_COL_ALIASES = ["fig_index", "figure_index", "fig_idx", "figure_idx",
                   "fig_num", "caption_index"]
CAPTION_ALIASES = ["caption", "caption_text", "text", "caption_prefix"]


class LoadedIndex:
    """Uniform search interface over a FAISS index or a raw numpy matrix."""

    def __init__(self, faiss_index=None, matrix=None):
        assert (faiss_index is None) != (matrix is None)
        self.faiss = faiss_index
        self.matrix = matrix
        self.n = faiss_index.ntotal if faiss_index is not None else len(matrix)

    def search(self, query_vecs, k):
        q = np.ascontiguousarray(query_vecs, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self.faiss is not None:
            return self.faiss.search(q, k)
        sims = q @ self.matrix.T
        ids = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, ids, axis=1), ids

    def vectors(self, row_ids):
        if self.faiss is not None:
            import faiss
            ids = np.asarray(row_ids, dtype=np.int64)
            if hasattr(self.faiss, "reconstruct_batch"):
                return self.faiss.reconstruct_batch(ids)
            return np.stack([self.faiss.reconstruct(int(i)) for i in ids])
        return np.asarray(self.matrix[np.asarray(row_ids)])


def _find_one(index_dir, patterns, what):
    hits = [p for pat in patterns for p in sorted(index_dir.glob(pat))]
    if not hits:
        raise FileNotFoundError(
            f"no {what} found in {index_dir} (looked for {patterns}); "
            f"contents: {[p.name for p in index_dir.iterdir()]}")
    if len(hits) > 1:
        print(f"  multiple {what} candidates {[h.name for h in hits]}, "
              f"using {hits[0].name}")
    return hits[0]


def load_caption_index(index_dir, captions_parquet=None):
    """Returns (LoadedIndex, meta DataFrame with columns
    arxiv_id, fig_index, title, caption -- row-aligned to the index)."""
    index_dir = Path(index_dir)

    info_path = index_dir / "info.json"
    if info_path.exists():
        print(f"info.json: {json.load(open(info_path))}")

    try:
        vec_path = _find_one(index_dir, ["*.faiss", "*.index"], "FAISS index")
        import faiss
        loaded = LoadedIndex(faiss_index=faiss.read_index(str(vec_path)))
        print(f"vectors: {vec_path.name} (FAISS, {loaded.n} rows)")
    except FileNotFoundError:
        vec_path = _find_one(index_dir, ["*.npy"], "embedding matrix")
        loaded = LoadedIndex(matrix=np.load(vec_path, mmap_mode="r"))
        print(f"vectors: {vec_path.name} (numpy, {loaded.n} rows)")

    meta_path = _find_one(index_dir, ["*.parquet", "*.jsonl"], "metadata")
    meta = pd.read_parquet(meta_path) if meta_path.suffix == ".parquet" \
        else pd.read_json(meta_path, lines=True)
    print(f"metadata: {meta_path.name}, columns {list(meta.columns)}")
    if len(meta) != loaded.n:
        raise ValueError(f"metadata rows ({len(meta)}) != index rows "
                         f"({loaded.n}); wrong file pair?")

    if "arxiv_id" not in meta.columns:
        raise ValueError(f"no arxiv_id column in {meta_path.name}")
    fig_col = next((c for c in FIG_COL_ALIASES if c in meta.columns), None)
    if fig_col is None:
        raise ValueError(f"no figure-index column in {meta_path.name} "
                         f"(tried {FIG_COL_ALIASES})")
    meta = meta.rename(columns={fig_col: "fig_index"})
    cap_col = next((c for c in CAPTION_ALIASES if c in meta.columns), None)
    if cap_col and cap_col != "caption":
        meta = meta.rename(columns={cap_col: "caption"})

    need_caption = "caption" not in meta.columns or cap_col == "caption_prefix"
    need_title = "title" not in meta.columns
    if need_caption or need_title:
        if captions_parquet is None:
            raise ValueError(
                "index metadata lacks full caption/title columns; pass "
                "captions_parquet to join them")
        src = pd.read_parquet(captions_parquet)
        if "captions" not in src.columns:
            raise ValueError(f"{captions_parquet} has no 'captions' column; "
                             f"columns: {list(src.columns)}")
        long = src[["arxiv_id", "title", "captions"]].explode("captions")
        # the index uses 1-based figure positions (generate_subscripts)
        long["fig_index"] = long.groupby("arxiv_id").cumcount() + 1
        long = long.rename(columns={"captions": "caption_full"})
        meta = meta.merge(long, on=["arxiv_id", "fig_index"], how="left",
                          suffixes=("", "_joined"))
        if need_caption:
            missing = int(meta.caption_full.isna().sum())
            if missing:
                print(f"  WARNING: {missing} rows had no caption after join")
            meta["caption"] = meta.caption_full.fillna("")
        if need_title:
            tcol = "title_joined" if "title_joined" in meta.columns else "title"
            meta["title"] = meta[tcol].fillna("")
        print(f"joined captions/titles from {captions_parquet}")

    meta["fig_index"] = meta.fig_index.astype(int)
    meta = meta[["arxiv_id", "fig_index", "title", "caption"]]
    return loaded, meta.reset_index(drop=True)
