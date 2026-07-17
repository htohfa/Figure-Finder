"""Embedding backends: OpenAI API or local sentence-transformers, with the
query/document prefixes each model family expects."""
import numpy as np

PREFIX_RULES = [
    ("bge-", {"query": "Represent this sentence for searching relevant passages: ",
              "doc": ""}),
    ("e5-", {"query": "query: ", "doc": "passage: "}),
    ("multilingual-e5", {"query": "query: ", "doc": "passage: "}),
    ("Qwen3-Embedding", {"query": "Instruct: Given a figure search query, retrieve the "
                                  "figure caption that matches it\nQuery: ",
                         "doc": ""}),
]


def prefixes_for(model_name):
    for needle, rules in PREFIX_RULES:
        if needle.lower() in model_name.lower():
            return rules
    return {"query": "", "doc": ""}


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model="text-embedding-3-small", dim=512):
        import os
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                             max_retries=0, timeout=60)
        self.model = model
        self.dim = dim

    def embed(self, texts, is_query=False, batch_size=128):
        import time
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(8):
                try:
                    resp = self.client.embeddings.create(
                        model=self.model, input=batch, dimensions=self.dim)
                    out.extend(d.embedding for d in resp.data)
                    break
                except Exception as e:
                    if "429" not in str(e) or attempt == 7:
                        raise
                    time.sleep(10 * (attempt + 1))
        X = np.array(out, dtype=np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def info(self):
        return {"backend": "openai", "model": self.model, "dim": self.dim}


class LocalEmbedder:
    name = "local"

    def __init__(self, model="BAAI/bge-base-en-v1.5", device=None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.model = SentenceTransformer(model, device=device)
        self.prefixes = prefixes_for(model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts, is_query=False, batch_size=64):
        prefix = self.prefixes["query" if is_query else "doc"]
        inputs = [prefix + t for t in texts]
        X = self.model.encode(inputs, batch_size=batch_size,
                              normalize_embeddings=True,
                              show_progress_bar=len(texts) > 1000)
        return np.asarray(X, dtype=np.float32)

    def info(self):
        return {"backend": "local", "model": self.model_name, "dim": self.dim}


def make_embedder(backend, model=None, dim=512):
    if backend == "openai":
        return OpenAIEmbedder(model=model or "text-embedding-3-small", dim=dim)
    if backend == "local":
        return LocalEmbedder(model=model or "BAAI/bge-base-en-v1.5")
    raise ValueError(f"unknown backend {backend}")


def embedder_from_info(info):
    if info["backend"] == "openai":
        return OpenAIEmbedder(model=info["model"], dim=info["dim"])
    return LocalEmbedder(model=info["model"])
