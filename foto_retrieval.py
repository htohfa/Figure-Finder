"""bge query embedding + FAISS search over the hosted foto-index."""
import numpy as np
import pandas as pd
from pathlib import Path

BGE_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX_REPO = "htohfa/foto-index"


class Retriever:
    def __init__(self, repo=INDEX_REPO):
        import faiss
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        local = Path(snapshot_download(repo, repo_type="dataset"))
        self.index = faiss.read_index(str(next(local.glob("*.faiss"))))
        meta = pd.read_parquet(next(local.glob("meta.parquet")))
        self.meta = meta.rename(columns={"fig_idx": "fig_index"}).reset_index(drop=True)
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def search(self, query, k=100):
        v = self.model.encode([BGE_PREFIX + query],
                              normalize_embeddings=True).astype(np.float32)
        sims, ids = self.index.search(v, k)
        out = []
        for s, i in zip(sims[0], ids[0]):
            if i < 0:
                continue
            row = self.meta.iloc[int(i)]
            out.append({"arxiv_id": str(row.arxiv_id),
                        "fig_index": int(row.fig_index),
                        "caption_prefix": str(row.caption),
                        "score": float(s)})
        return out
