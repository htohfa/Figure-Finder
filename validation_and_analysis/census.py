"""Pull the astro-ph text slice out of ArxivCap via DuckDB (no image download)."""
import argparse
import os
import time
from collections import Counter
from pathlib import Path

import duckdb

SRC = "hf://datasets/MMInstruction/ArxivCap/data/*.parquet"

EXTRACT = """
SELECT
    arxiv_id,
    title,
    abstract,
    meta.meta_from_kaggle.categories AS categories,
    meta.meta_from_s2.citationCount AS citations,
    list_transform(caption_images, x -> x.caption) AS captions,
    list_transform(
        caption_images,
        x -> substr(
            array_to_string(
                flatten(list_transform(x.cil_pairs, y -> y.image_ocr)), ' '
            ), 1, 500)
    ) AS ocr
FROM read_parquet('{src}')
WHERE contains(meta.meta_from_kaggle.categories, 'astro-ph')
{limit}
"""


def year_from_arxiv_id(arxiv_id):
    try:
        yy = int(arxiv_id.split("/")[1][:2]) if "/" in arxiv_id else int(arxiv_id[:2])
        return 1900 + yy if yy > 80 else 2000 + yy
    except (ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="astro_captions.parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parts-dir", default="census_parts")
    args = ap.parse_args()

    con = duckdb.connect()
    for setting in ["SET enable_progress_bar = true", "SET http_retries = 8",
                    "SET http_retry_wait_ms = 2000", "SET http_retry_backoff = 2",
                    "SET threads = 2"]:
        try:
            con.execute(setting)
        except duckdb.Error:
            pass

    token = os.environ.get("HF_TOKEN", "")
    if token:
        con.execute(f"CREATE SECRET hf (TYPE HUGGINGFACE, TOKEN '{token}')")
    else:
        print("WARNING: no HF_TOKEN set, anonymous requests get rate limited fast.")

    print("Extracting astro-ph slice (text columns only)...")
    if args.limit or not args.src.startswith("hf://"):
        limit = f"LIMIT {args.limit}" if args.limit else ""
        con.execute(f"COPY ({EXTRACT.format(src=args.src, limit=limit)}) "
                    f"TO '{args.out}' (FORMAT PARQUET)")
    else:
        parts = Path(args.parts_dir)
        parts.mkdir(exist_ok=True)
        chunks = [f"{y:02d}" for y in list(range(91, 100)) + list(range(0, 24))]
        for yy in chunks:
            part_file = parts / f"astro_{yy}.parquet"
            if part_file.exists():
                continue
            chunk_src = args.src.replace("*", f"arXiv_src_{yy}*")
            query = EXTRACT.format(src=chunk_src, limit="")
            for attempt in range(5):
                try:
                    con.execute(f"COPY ({query}) TO '{part_file}' (FORMAT PARQUET)")
                    print(f"  {yy}: done")
                    break
                except duckdb.Error as e:
                    msg = str(e)
                    if "No files found" in msg:
                        print(f"  {yy}: no files, skipping")
                        break
                    if attempt == 4:
                        raise
                    wait = 30 * (attempt + 1)
                    print(f"  {yy}: {msg[:80]} -- retrying in {wait}s")
                    time.sleep(wait)
        con.execute(f"COPY (SELECT * FROM read_parquet('{parts}/astro_*.parquet')) "
                    f"TO '{args.out}' (FORMAT PARQUET)")

    papers, figures, usable, fpp, cap_chars = con.execute(f"""
        SELECT count(*),
               sum(len(captions)),
               sum(len(list_filter(captions, c -> length(c) >= 20))),
               avg(len(captions)),
               round(avg(list_aggregate(list_transform(captions, c -> length(c)), 'avg')), 0)
        FROM read_parquet('{args.out}')
    """).fetchone()

    print(f"\nPapers: {papers:,}  Figures: {int(figures):,}  "
          f"Usable captions: {int(usable):,}  Figs/paper: {fpp:.1f}  "
          f"Mean caption: {int(cap_chars)} chars")

    ids = [r[0] for r in con.execute(
        f"SELECT arxiv_id FROM read_parquet('{args.out}')").fetchall()]
    by_year = Counter(y for y in (year_from_arxiv_id(i) for i in ids) if y)
    for y in sorted(by_year):
        print(f"  {y}: {by_year[y]:,}")
    print(f"\nSlice written to {args.out}")


if __name__ == "__main__":
    main()
