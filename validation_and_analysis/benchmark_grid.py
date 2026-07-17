"""FOTO vs pathfinder vs Semantic Scholar over 5 query registers x 4 query forms (Table 4 data)."""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

from index_loader import load_caption_index

STYLES = ["terse", "casual", "vague", "detailed", "notation"]
HAIKU = "claude-haiku-4-5-20251001"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENAI_EMB_URL = "https://api.openai.com/v1/embeddings"
PAPER_K = 50    # paper retrieval depth for baselines and paper_rank cap
FIG_K = 200     # caption retrieval depth for ours


# ----------------------------------------------------------------- LLM

def anthropic_call(prompt, max_tokens=2000, retries=4):
    key = os.environ["ANTHROPIC_API_KEY"]
    body = {"model": HAIKU, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    for attempt in range(retries):
        r = requests.post(ANTHROPIC_URL, headers=headers, json=body,
                          timeout=120)
        if r.status_code == 200:
            return "".join(b.get("text", "") for b in r.json()["content"])
        if r.status_code in (429, 500, 529):
            time.sleep(2 ** attempt * 2)
            continue
        raise RuntimeError(f"anthropic {r.status_code}: {r.text[:300]}")
    raise RuntimeError("anthropic: retries exhausted")


def parse_json_block(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


QUERY_PROMPT = """You are generating benchmark search queries for a figure \
retrieval system. Below is the title of an astronomy paper and the caption \
of one figure from it. Write five queries a researcher might type when \
looking for this exact figure, WITHOUT having the caption in front of them:

1. "terse": short expert phrasing, 4-10 words, standard jargon.
2. "casual": conversational, as if asking a colleague, one sentence.
3. "vague": the researcher half-remembers the figure. Keep it faithful to \
the figure's actual content but drop specifics (no survey names, no exact \
quantities); 5-15 words.
4. "detailed": two sentences describing axes, curves, and what is compared.
5. "notation": uses symbols/notation an expert would use (e.g. sigma_8, \
z~2, P(k)).

Do not copy phrases longer than 4 words from the caption. Respond with ONLY \
a JSON object: {{"terse": "...", "casual": "...", "vague": "...", \
"detailed": "...", "notation": "..."}}

Title: {title}
Caption: {caption}"""


VARIANTS_PROMPT = """You produce two variants of search queries used to \
retrieve one SPECIFIC scientific figure by matching its caption text.

For each query below produce:

"expanded": an expanded version under strict rules:
- Add only concrete, discriminative terms the target caption would \
plausibly contain, inferred strictly from what the query already says: \
standard synonyms, the conventional name of the plot type or statistic, \
standard notation for quantities the query names.
- Do NOT add survey names, instruments, wavelengths, redshifts, numeric \
values, or subfield context unless the query itself mentions or directly \
implies them.
- Do NOT generalize: the expansion must not match a broader class of \
figures than the original query does.
- Keep the original wording inside the expansion. Max 40 words.

"keywords": 3-8 keyword terms extracted from the query, space-separated, \
no stopwords, no sentence structure, no terms not present in or directly \
implied by the query.

Respond with ONLY a JSON array of {n} objects: \
[{{"i": 0, "expanded": "...", "keywords": "..."}}, ...] with "i" echoing \
the input index.

Queries:
{queries}"""


def generate_queries(title, caption):
    out = parse_json_block(anthropic_call(
        QUERY_PROMPT.format(title=title, caption=caption[:1500])))
    return {s: str(out[s]) for s in STYLES}


def generate_variants(queries):
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(queries))
    out = parse_json_block(anthropic_call(
        VARIANTS_PROMPT.format(n=len(queries), queries=numbered)))
    exp, kw = list(queries), list(queries)
    for item in out:
        i = int(item["i"])
        if item.get("expanded"):
            exp[i] = str(item["expanded"])
        if item.get("keywords"):
            kw[i] = str(item["keywords"])
    return exp, kw


# ----------------------------------------------------------- external APIs

class S2Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.d = {}
        if self.path.exists():
            for line in self.path.open():
                rec = json.loads(line)
                self.d[rec["q"]] = rec["ids"]
        self.f = self.path.open("a")

    def get(self, q):
        return self.d.get(q)

    def put(self, q, ids):
        self.d[q] = ids
        self.f.write(json.dumps({"q": q, "ids": ids}) + "\n")
        self.f.flush()


def s2_top(query, api_key, cache, limit=PAPER_K, throttle=1.5, retries=6):
    """Returns None iff the API never gave a valid response.
    An empty list is a valid 'found nothing' outcome."""
    hit = cache.get(query)
    if hit is not None:
        return hit
    params = {"query": query, "fields": "externalIds", "limit": limit}
    headers = {"x-api-key": api_key}
    for attempt in range(retries):
        r = requests.get(S2_URL, params=params, headers=headers, timeout=60)
        time.sleep(throttle)
        if r.status_code == 200:
            ids = [(p.get("externalIds") or {}).get("ArXiv")
                   for p in r.json().get("data", []) or []]
            ids = [i for i in ids if i]
            cache.put(query, ids)
            return ids
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        return None
    return None


def openai_embed(texts, retries=4):
    key = os.environ["OPENAI_API_KEY"]
    headers = {"Authorization": f"Bearer {key}",
               "content-type": "application/json"}
    body = {"model": "text-embedding-3-small", "input": texts}
    for attempt in range(retries):
        r = requests.post(OPENAI_EMB_URL, headers=headers, json=body,
                          timeout=120)
        if r.status_code == 200:
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            v = np.array([d["embedding"] for d in data], dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        if r.status_code in (429, 500):
            time.sleep(2 ** attempt * 2)
            continue
        raise RuntimeError(f"openai {r.status_code}: {r.text[:300]}")
    raise RuntimeError("openai: retries exhausted")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--captions-parquet", default="astro_captions.parquet")
    ap.add_argument("--pathfinder-emb", default=None)
    ap.add_argument("--pathfinder-meta", default=None)
    ap.add_argument("--n-figures", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_grid")
    ap.add_argument("--s2-forms", default="orig,expanded,keywords")
    ap.add_argument("--s2-throttle", type=float, default=1.5)
    args = ap.parse_args()

    loaded, meta = load_caption_index(args.index_dir, args.captions_parquet)
    if not meta.index.equals(pd.RangeIndex(len(meta))):
        meta = meta.reset_index(drop=True)
    print(f"index: {len(meta)} captions, {meta.arxiv_id.nunique()} papers")

    do_pf = args.pathfinder_emb and args.pathfinder_meta
    if do_pf:
        pf_emb = np.load(args.pathfinder_emb)
        pf_meta = pd.read_parquet(args.pathfinder_meta)
        assert len(pf_meta) == len(pf_emb), "pathfinder corpus misaligned"
        pf_id_set = set(pf_meta.arxiv_id.astype(str))
    else:
        print("WARNING: no pathfinder corpus given, skipping pathfinder")

    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    s2_forms = [f.strip() for f in args.s2_forms.split(",") if f.strip()]
    if not s2_key:
        print("WARNING: SEMANTIC_SCHOLAR_API_KEY unset, skipping s2")
    s2_cache = S2Cache(f"{args.out}_s2_cache.jsonl") if s2_key else None

    bge = SentenceTransformer("BAAI/bge-base-en-v1.5")

    rng = random.Random(args.seed)
    ok = np.flatnonzero(
        meta.caption.str.len().ge(30).fillna(False).to_numpy()).tolist()
    n_fig = min(args.n_figures, len(ok))
    if n_fig < args.n_figures:
        print(f"WARNING: only {len(ok)} usable captions, sampling {n_fig}")
    sample = rng.sample(ok, n_fig)

    out_jsonl = Path(f"{args.out}.jsonl")
    done = set()
    if out_jsonl.exists():
        for line in out_jsonl.open():
            rec = json.loads(line)
            done.add((rec["arxiv_id"], rec["fig_index"]))
        print(f"resuming: {len(done)} figures already done")

    by_paper = meta.groupby("arxiv_id").indices
    arxiv_ids = meta.arxiv_id.to_numpy()

    # --------------------------------------------------------- scoring

    def rank_of(seq, target):
        for i, x in enumerate(seq):
            if x == target:
                return i + 1
        return None

    def fig_rank_in(ranked_rows, target_row):
        ranked_rows = np.asarray(ranked_rows)
        pos = np.where(ranked_rows == target_row)[0]
        return int(pos[0]) + 1 if len(pos) else None

    def distinct_paper_order(ranked_rows, cap=PAPER_K):
        seen, order = set(), []
        for pid in arxiv_ids[np.asarray(ranked_rows, dtype=int)]:
            if pid not in seen:
                seen.add(pid)
                order.append(pid)
            if len(order) >= cap:
                break
        return order

    def ours_cell(qvec, target_row, target_paper):
        _, ids = loaded.search(qvec, FIG_K)
        top = ids[0]
        return {"fig_rank": fig_rank_in(top, target_row),
                "paper_rank": rank_of(distinct_paper_order(top),
                                      target_paper)}

    def ours_fused_cell(v_orig, v_exp, target_row, target_paper, k=FIG_K):
        _, ids_a = loaded.search(v_orig, k)
        _, ids_b = loaded.search(v_exp, k)
        union = np.unique(np.concatenate([ids_a[0], ids_b[0]]))
        vecs = loaded.vectors(union)
        fused = np.maximum(vecs @ v_orig, vecs @ v_exp)
        top = union[np.argsort(-fused)][:k]
        return {"fig_rank": fig_rank_in(top, target_row),
                "paper_rank": rank_of(distinct_paper_order(top),
                                      target_paper)}

    def rank_pool(query_vec, rows):
        if len(rows) == 0:
            return []
        rows = np.asarray(rows)
        sims = loaded.vectors(rows) @ query_vec
        return list(rows[np.argsort(-sims)])

    def pooled_cell(paper_order, rank_vec, target_row, target_paper):
        pool = [r for pid in paper_order for r in by_paper.get(pid, [])]
        ranked = rank_pool(rank_vec, pool)
        return {"fig_rank": fig_rank_in(ranked, target_row),
                "paper_rank": rank_of(paper_order, target_paper),
                "pool_size": len(pool)}

    def pf_paper_order(qvec_openai, k=PAPER_K):
        sims = pf_emb @ qvec_openai
        return pf_meta.iloc[np.argsort(-sims)[:k]].arxiv_id.tolist()

    def pf_fused_paper_order(v_orig, v_exp, k=PAPER_K):
        fused = np.maximum(pf_emb @ v_orig, pf_emb @ v_exp)
        return pf_meta.iloc[np.argsort(-fused)[:k]].arxiv_id.tolist()

    # ------------------------------------------------------------- loop

    with out_jsonl.open("a") as fout:
        n_done = 0
        for row_i in sample:
            row = meta.iloc[row_i]
            key = (row.arxiv_id, int(row.fig_index))
            if key in done:
                continue
            try:
                qs = generate_queries(row.title, row.caption)
                q_orig = [qs[s] for s in STYLES]
                q_exp, q_kw = generate_variants(q_orig)
            except Exception as e:
                print(f"skip {key}: query gen failed ({e})")
                continue

            forms_text = {"orig": q_orig, "expanded": q_exp,
                          "keywords": q_kw}
            forms_bge = {f: bge.encode(t, normalize_embeddings=True)
                         for f, t in forms_text.items()}
            if do_pf:
                try:
                    flat = openai_embed(q_orig + q_exp + q_kw)
                except Exception as e:
                    print(f"skip {key}: openai embed failed ({e})")
                    continue
                ns = len(STYLES)
                forms_pf = {"orig": flat[:ns], "expanded": flat[ns:2 * ns],
                            "keywords": flat[2 * ns:]}

            rec = {"arxiv_id": row.arxiv_id, "fig_index": int(row.fig_index),
                   "queries": qs,
                   "expanded": dict(zip(STYLES, q_exp)),
                   "keywords": dict(zip(STYLES, q_kw)),
                   "target_in_pf": bool(str(row.arxiv_id) in pf_id_set)
                   if do_pf else None,
                   "cells": {}}

            for si, style in enumerate(STYLES):
                cells = {}

                for form in ("orig", "expanded", "keywords"):
                    v = forms_bge[form][si]
                    cells[f"ours/{form}"] = ours_cell(v, row_i, row.arxiv_id)

                    if do_pf:
                        order = pf_paper_order(forms_pf[form][si])
                        cells[f"pathfinder/{form}"] = pooled_cell(
                            order, v, row_i, row.arxiv_id)

                    if s2_key and form in s2_forms:
                        ids = s2_top(forms_text[form][si], s2_key, s2_cache,
                                     throttle=args.s2_throttle)
                        if ids is None:
                            pass  # API failure: cell absent
                        else:
                            c = pooled_cell(ids, v, row_i, row.arxiv_id)
                            c["n_arxiv_ids"] = len(ids)
                            cells[f"s2/{form}"] = c

                cells["ours/fused"] = ours_fused_cell(
                    forms_bge["orig"][si], forms_bge["expanded"][si],
                    row_i, row.arxiv_id)
                if do_pf:
                    order = pf_fused_paper_order(
                        forms_pf["orig"][si], forms_pf["expanded"][si])
                    cells["pathfinder/fused"] = pooled_cell(
                        order, forms_bge["orig"][si], row_i, row.arxiv_id)

                rec["cells"][style] = cells

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                print(f"{n_done} new figures done "
                      f"({n_done + len(done)}/{len(sample)} total)")

    n_total = sum(1 for _ in out_jsonl.open())
    print(f"\ndone: {n_total} figures in {out_jsonl}")
    print(f"summarize with:  python summarize.py {out_jsonl}")


if __name__ == "__main__":
    sys.exit(main())
