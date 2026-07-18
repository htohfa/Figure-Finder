"""On-demand figure fetch from ArxivCap on HuggingFace, with a local cache.

Each figure is fetched at most once: the JPEG and a sidecar json holding the
paper title and full caption are written to the cache dir, keyed by
(arxiv_id, fig_index). Misses are grouped by ArxivCap month chunk so one
query's candidates cost one DuckDB range read per chunk.
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

HF_SRC = "hf://datasets/MMInstruction/ArxivCap/data"


def yymm_of(arxiv_id):
    s = str(arxiv_id)
    if "/" in s:
        return re.sub(r"\D", "", s.split("/", 1)[1])[:4]
    return s[:4]


def safe_stem(arxiv_id, fig_index):
    return re.sub(r"[^0-9a-zA-Z.]+", "_", str(arxiv_id)) + f"_f{fig_index}"


def _cached(cache, a, f):
    stem = safe_stem(a, f)
    img, meta = cache / f"{stem}.jpg", cache / f"{stem}.json"
    if img.exists() and meta.exists():
        d = json.loads(meta.read_text())
        d["image_path"] = str(img)
        return d
    return None


def fetch_figures(needed, cache_dir="figure_cache", log=None):
    """needed: iterable of (arxiv_id, fig_index 1-based).
    Returns {(a, f): {"image_path", "title", "caption"}} for what was found."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out, todo = {}, set()
    for a, f in needed:
        hit = _cached(cache, a, f)
        if hit:
            out[(a, f)] = hit
        else:
            todo.add((a, f))
    if not todo:
        return out

    import duckdb
    con = duckdb.connect()
    for s in ["SET http_retries = 6", "SET http_retry_wait_ms = 1500",
              "SET threads = 2"]:
        try:
            con.execute(s)
        except Exception:
            pass
    token = os.environ.get("HF_TOKEN", "")
    if token:
        con.execute(f"CREATE SECRET hf (TYPE HUGGINGFACE, TOKEN '{token}')")

    by_chunk = defaultdict(set)
    for a, f in todo:
        by_chunk[yymm_of(a)].add((a, f))
    for yymm, pairs in sorted(by_chunk.items()):
        if log:
            log(f"  fetching {len(pairs)} figure(s) from chunk {yymm}...")
        ids = sorted({a for a, _ in pairs})
        idlist = ",".join("'" + i.replace("'", "''") + "'" for i in ids)
        src = f"{HF_SRC}/arXiv_src_{yymm}_*.parquet"
        try:
            rows = con.execute(
                f"SELECT arxiv_id, title, caption_images "
                f"FROM read_parquet('{src}') "
                f"WHERE arxiv_id IN ({idlist})").fetch_arrow_table().to_pylist()
        except Exception as e:
            if log:
                log(f"  chunk {yymm} failed: {str(e)[:80]}")
            continue
        for row in rows:
            a = row["arxiv_id"]
            caps = row["caption_images"] or []
            for (aa, f) in pairs:
                if aa != a or f > len(caps):
                    continue
                entry = caps[f - 1] or {}
                cil = entry.get("cil_pairs") or []
                img = (cil[0] or {}).get("image") if cil else None
                data = img.get("bytes") if isinstance(img, dict) else img
                if not data:
                    continue
                stem = safe_stem(a, f)
                (cache / f"{stem}.jpg").write_bytes(data)
                d = {"title": row.get("title") or "",
                     "caption": entry.get("caption") or ""}
                (cache / f"{stem}.json").write_text(json.dumps(d))
                d["image_path"] = str(cache / f"{stem}.jpg")
                out[(a, f)] = d
    return out
