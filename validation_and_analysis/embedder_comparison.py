"""bge (with query prefix) vs OpenAI embedding comparison on the benchmark queries (Table 3)."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from index_loader import load_caption_index
from embedders import embedder_from_info, LocalEmbedder

STYLES = ["terse", "casual", "vague", "detailed", "notation"]
KS = [1, 5, 20, 50]


def eval_index(loaded, meta, embed_fn, records, k=200):
    pos = {(a, int(f)): i for i, (a, f) in
           enumerate(zip(meta.arxiv_id, meta.fig_index))}
    out = {}
    for style in STYLES:
        queries = [r["queries"][style] for r in records]
        targets = [(r["arxiv_id"], int(r["fig_index"])) for r in records]
        Q = embed_fn(queries)
        ranks = []
        for qi, tgt in enumerate(targets):
            tgt_row = pos.get(tgt)
            if tgt_row is None:
                ranks.append(None)
                continue
            _, ids = loaded.search(Q[qi], k)
            hit = np.where(ids[0] == tgt_row)[0]
            ranks.append(int(hit[0]) + 1 if len(hit) else None)
        out[style] = ranks
        print(f"    {style}: done")
    return out


def recall_rows(name, ranks_by_style):
    rows = []
    for style in STYLES:
        rs = ranks_by_style[style]
        row = {"index": name, "style": style, "n": len(rs)}
        for k in KS:
            row[f"R@{k}"] = round(
                sum(1 for r in rs if r is not None and r <= k) / len(rs), 3)
        rows.append(row)
    mean = {"index": name, "style": "MEAN", "n": rows[0]["n"]}
    for k in KS:
        mean[f"R@{k}"] = round(np.mean([r[f"R@{k}"] for r in rows]), 3)
    return rows + [mean]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_grid.jsonl")
    ap.add_argument("--index-bge", default="indexes/title_caption_bge-base-en-v1.5")
    ap.add_argument("--index-openai", default="indexes/title_caption")
    ap.add_argument("--captions-parquet", default="astro_captions.parquet")
    ap.add_argument("--out", default="embedder_comparison")
    ap.add_argument("--include-noprefix", action="store_true",
                    help="also evaluate bge without its query prefix")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.results)]
    print(f"{len(records)} figures in {args.results}")
    all_rows = []

    loaded, meta = load_caption_index(args.index_bge, args.captions_parquet)
    bge = LocalEmbedder("BAAI/bge-base-en-v1.5")
    print("  bge-base-en-v1.5, with query prefix:")
    all_rows += recall_rows("bge_prefixed", eval_index(
        loaded, meta, lambda q: bge.embed(q, is_query=True), records))
    if args.include_noprefix:
        print("  bge-base-en-v1.5, without prefix:")
        all_rows += recall_rows("bge_noprefix", eval_index(
            loaded, meta, lambda q: bge.embed(q, is_query=False), records))

    if args.index_openai and os.environ.get("OPENAI_API_KEY"):
        loaded_oa, meta_oa = load_caption_index(args.index_openai,
                                                args.captions_parquet)
        info = json.load(open(Path(args.index_openai) / "info.json"))
        oa = embedder_from_info(info)
        print(f"  {info['model']} ({info['dim']}d):")
        all_rows += recall_rows("openai_small", eval_index(
            loaded_oa, meta_oa, lambda q: oa.embed(q, is_query=True), records))
    else:
        print("  skipping openai index (missing path or OPENAI_API_KEY)")

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{args.out}.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {args.out}.csv")


if __name__ == "__main__":
    main()
