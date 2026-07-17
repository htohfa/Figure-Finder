"""Embed the caption slice and build a FAISS index (openai or local bge backend)."""
import argparse
import json
import os
import re
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embedders import make_embedder

MIN_CAPTION_CHARS = 20
PART_SIZE = 20000
REQUEST_BATCH = 256
MAX_CHARS_PER_TEXT = 6000


def clean_latex(text):
    text = re.sub(r"\\cite[pt]?\*?(\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\(ref|eqref|label)\{[^}]*\}", "", text)
    text = re.sub(r"\\(rm|it|bf|mathrm|mathcal|mathbf|text|textit|textbf)\b", "", text)
    text = text.replace("$", " ").replace("\\", " ")
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"_\s+", "_", text)
    return re.sub(r"\s+", " ", text).strip()


def load_rows(slice_path, strategy, limit=None):
    con = duckdb.connect()
    if strategy == "abstract":
        q = f"""
        SELECT arxiv_id, -1 AS fig_idx, title, abstract, '' AS caption, citations
        FROM read_parquet('{slice_path}') WHERE length(abstract) >= 50
        """
    else:
        q = f"""
        SELECT arxiv_id, u.fig_idx, title, abstract, u.caption, citations
        FROM read_parquet('{slice_path}'),
             LATERAL (SELECT unnest(captions) AS caption,
                             generate_subscripts(captions, 1) AS fig_idx) AS u
        WHERE length(u.caption) >= {MIN_CAPTION_CHARS}
        """
    if limit:
        q += f" LIMIT {limit}"
    result = con.execute(q)
    try:
        return result.to_arrow_table()
    except AttributeError:
        return result.fetch_arrow_table()


def build_text(row, strategy):
    caption = clean_latex(row["caption"]) if row.get("caption") else ""
    if strategy == "caption":
        text = caption
    elif strategy == "title_caption":
        text = f"{row['title']} | {caption}"
    elif strategy == "concat":
        text = f"{row['title']} | {row['abstract']} | {caption}"
    elif strategy == "abstract":
        text = f"{row['title']} | {row['abstract']}"
    else:
        raise ValueError(strategy)
    return text[:MAX_CHARS_PER_TEXT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="astro_captions.parquet")
    ap.add_argument("--strategy", required=True,
                    choices=["caption", "title_caption", "concat", "abstract"])
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--backend", default="openai", choices=["openai", "local"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="indexes")
    args = ap.parse_args()

    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (or use --backend local)")
    embedder = make_embedder(args.backend, args.model, args.dim)

    suffix = args.suffix if args.suffix is not None else (
        "" if args.backend == "openai"
        else "_" + embedder.info()["model"].split("/")[-1])
    out_dir = Path(args.out) / (args.strategy + suffix)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.slice, args.strategy, args.limit).to_pylist()
    print(f"{len(rows):,} rows to embed for strategy '{args.strategy}'")

    done_parts = sorted(parts_dir.glob("part_*.parquet"))
    done_rows = sum(pq.read_metadata(p).num_rows for p in done_parts)
    if done_rows:
        print(f"Resuming: {done_rows:,} rows already embedded")

    part_idx = len(done_parts)
    buffer_rows, buffer_vecs = [], []
    for start in range(done_rows, len(rows), REQUEST_BATCH):
        batch = rows[start:start + REQUEST_BATCH]
        texts = [build_text(r, args.strategy) for r in batch]
        buffer_rows.extend(batch)
        buffer_vecs.append(embedder.embed(texts, is_query=False))
        if sum(v.shape[0] for v in buffer_vecs) >= PART_SIZE \
                or start + REQUEST_BATCH >= len(rows):
            vec_arr = np.vstack(buffer_vecs)
            pq.write_table(pa.table({
                "arxiv_id": [r["arxiv_id"] for r in buffer_rows],
                "fig_idx": [r["fig_idx"] for r in buffer_rows],
                "caption": [r["caption"][:200] for r in buffer_rows],
                "citations": [r["citations"] for r in buffer_rows],
                "embedding": list(vec_arr),
            }), parts_dir / f"part_{part_idx:04d}.parquet")
            print(f"  wrote part {part_idx} ({start + len(batch):,}/{len(rows):,})")
            part_idx += 1
            buffer_rows, buffer_vecs = [], []

    import faiss
    metas, vecs = [], []
    for p in sorted(parts_dir.glob("part_*.parquet")):
        t = pq.read_table(p)
        metas.append(t.drop_columns(["embedding"]))
        vecs.append(np.vstack(t.column("embedding").to_pylist()).astype(np.float32))
    meta = pa.concat_tables(metas)
    X = np.vstack(vecs)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    pq.write_table(meta, out_dir / "meta.parquet")
    json.dump(embedder.info(), (out_dir / "info.json").open("w"))
    print(f"Index built: {index.ntotal:,} vectors, dim {X.shape[1]} -> {out_dir}")


if __name__ == "__main__":
    main()
