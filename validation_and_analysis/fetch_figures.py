"""Fetch specific figure images from ArxivCap on HuggingFace via DuckDB range requests."""
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


def safe_name(arxiv_id, fig_index):
    return re.sub(r"[^0-9a-zA-Z.]+", "_", str(arxiv_id)) + f"_f{fig_index}.jpg"


def fetch_images(needed, image_dir, hf_src=HF_SRC):
    """needed: set of (arxiv_id, fig_index 1-based). Skips files already on disk."""
    import duckdb
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    todo = {(a, f) for (a, f) in needed
            if not (image_dir / safe_name(a, f)).exists()}
    if not todo:
        print("  all images already cached")
        return
    by_chunk = defaultdict(set)
    for a, f in todo:
        by_chunk[yymm_of(a)].add((a, f))
    print(f"  fetching {len(todo)} figures from {len(by_chunk)} ArxivCap chunks")

    con = duckdb.connect()
    for setting in ["SET http_retries = 8", "SET http_retry_wait_ms = 2000",
                    "SET threads = 2"]:
        try:
            con.execute(setting)
        except Exception:
            pass
    token = os.environ.get("HF_TOKEN", "")
    if token:
        con.execute(f"CREATE SECRET hf (TYPE HUGGINGFACE, TOKEN '{token}')")
    else:
        print("  WARNING: no HF_TOKEN; anonymous HF requests rate-limit fast")

    for yymm, pairs in sorted(by_chunk.items()):
        ids = sorted({a for a, _ in pairs})
        idlist = ",".join("'" + i.replace("'", "''") + "'" for i in ids)
        src = f"{hf_src}/arXiv_src_{yymm}_*.parquet"
        try:
            tab = con.execute(
                f"SELECT arxiv_id, caption_images FROM read_parquet('{src}') "
                f"WHERE arxiv_id IN ({idlist})").fetch_arrow_table()
        except Exception as e:
            print(f"  chunk {yymm}: fetch failed ({str(e)[:120]}), skipping")
            continue
        got = 0
        for row in tab.to_pylist():
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
                (image_dir / safe_name(a, f)).write_bytes(data)
                got += 1
        print(f"  chunk {yymm}: {got} figures saved")


if __name__ == "__main__":
    import argparse
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("ids_csv", help="csv with columns arxiv_id,fig_index")
    ap.add_argument("--image-dir", default="vision_images")
    args = ap.parse_args()
    needed = {(r["arxiv_id"], int(r["fig_index"]))
              for r in csv.DictReader(open(args.ids_csv))}
    fetch_images(needed, Path(args.image_dir))
