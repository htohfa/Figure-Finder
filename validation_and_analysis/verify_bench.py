"""Verification benchmark: vision (caption+figure) vs caption-only text, Sonnet + DeepSeek judges."""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse, base64, io, json, random, re, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from index_loader import load_caption_index
from embedders import LocalEmbedder

SONNET = "claude-sonnet-4-6"
TAUS = [0.0, 0.5, 0.8]
PROMPT_VER = "p2"   # bump when VERIFY_PROMPT changes; keys cached per version

VERIFY_PROMPT = """A researcher is searching for a published scientific \
figure with this query:

"{query}"

{materials_line}

Caption: {caption}

Judge the candidate against what the query actually specifies. If the query \
is broad or vague, any figure genuinely showing what it describes counts as \
a match; do not demand details the query never mentions. If the query \
specifies particulars (quantities, axes, plot type, what is compared), the \
figure must show them. Science match is the dominant factor: a figure on \
the wrong scientific topic should never score above 0.4.

Respond JSON only:
{{"match": true|false, "confidence": 0.0-1.0, \
"what_is_plotted": "<one sentence>", "reason": "<one sentence>"}}"""

MATERIALS_VISION = "Below is the candidate figure image together with its caption."
MATERIALS_TEXT = "Only the figure's caption is available (no image)."


from fetch_figures import fetch_images, safe_name


def image_b64(path, max_dim=1024):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode()


def parse_verdict(text):
    obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
    return {"match": bool(obj.get("match")),
            "confidence": float(obj.get("confidence", 0.0)),
            "reason": str(obj.get("reason", ""))[:200]}


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.d = {}
        if self.path.exists():
            for line in self.path.open():
                r = json.loads(line)
                self.d[r["key"]] = r["verdict"]
        self.f = self.path.open("a")

    def get(self, key):
        return self.d.get(key)

    def put(self, key, verdict):
        self.d[key] = verdict
        self.f.write(json.dumps({"key": key, "verdict": verdict}) + "\n")
        self.f.flush()


class AnthropicJudge:
    def __init__(self, model):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def judge(self, prompt, img_path=None, retries=5):
        content = []
        if img_path is not None:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": image_b64(img_path)}})
        content.append({"type": "text", "text": prompt})
        for attempt in range(retries):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=250,
                    messages=[{"role": "user", "content": content}])
                return parse_verdict(resp.content[0].text)
            except Exception as e:
                if attempt == retries - 1:
                    print(f"    judge failed: {str(e)[:100]}")
                    return None
                time.sleep(5 * (attempt + 1))


class DeepSeekJudge:
    """Text-only. OpenAI-compatible endpoint."""

    def __init__(self, model):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                             base_url="https://api.deepseek.com")
        self.model = model

    def judge(self, prompt, img_path=None, retries=5):
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, max_tokens=250,
                    messages=[{"role": "user", "content": prompt}])
                return parse_verdict(resp.choices[0].message.content)
            except Exception as e:
                if attempt == retries - 1:
                    print(f"    deepseek failed: {str(e)[:100]}")
                    return None
                time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_grid.jsonl")
    ap.add_argument("--index-bge",
                    default="indexes/title_caption_bge-base-en-v1.5")
    ap.add_argument("--captions-parquet", default="astro_captions.parquet")
    ap.add_argument("--image-dir", default="vision_images")
    ap.add_argument("--out", default="verify2")
    # MUST match the overnight run so the image cache is reused
    ap.add_argument("--registers", default="detailed,vague")
    ap.add_argument("--n-per-register", type=int, default=50)
    ap.add_argument("--verify-k", type=int, default=50)
    ap.add_argument("--vision-seed", type=int, default=123)
    ap.add_argument("--deepseek-model", default="deepseek-chat",
                    help="check this matches your DeepSeek account's model id")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.results)]
    registers = [r.strip() for r in args.registers.split(",")]
    rng = random.Random(args.vision_seed)
    sample = rng.sample(records, min(args.n_per_register * 2, len(records)))
    tasks = [(reg, rec) for reg in registers
             for rec in sample[:args.n_per_register]]
    print(f"{len(tasks)} queries "
          f"({args.n_per_register} figures x {registers})")

    loaded, meta = load_caption_index(args.index_bge, args.captions_parquet)
    if not meta.index.equals(pd.RangeIndex(len(meta))):
        meta = meta.reset_index(drop=True)
    bge = LocalEmbedder("BAAI/bge-base-en-v1.5")

    # RAW captions (no LaTeX stripping) for the judges
    import duckdb
    captions = {}
    for aid, caps in duckdb.connect().execute(
            f"SELECT arxiv_id, captions FROM "
            f"read_parquet('{args.captions_parquet}')").fetchall():
        for i, cap in enumerate(caps or []):
            captions[(aid, i + 1)] = cap or ""
    max_cap = max((len(c) for c in captions.values() if c), default=0)
    if max_cap <= 200:
        raise SystemExit("ABORT: captions look truncated; check the slice.")
    print(f"raw captions loaded (max {max_cap} chars)")

    # retrieval plans, identical to overnight run (bge prefixed)
    plans = []
    for reg, rec in tasks:
        q = rec["queries"][reg]
        v = bge.embed([q], is_query=True)[0]
        _, ids = loaded.search(v, args.verify_k)
        cands = [(meta.iloc[int(i)].arxiv_id, int(meta.iloc[int(i)].fig_index))
                 for i in ids[0]]
        target = (rec["arxiv_id"], int(rec["fig_index"]))
        plans.append({"register": reg, "query": q, "target": target,
                      "cands": cands,
                      "ret_rank": cands.index(target) + 1
                      if target in cands else None})

    image_dir = Path(args.image_dir)
    fetch_images({c for p in plans for c in p["cands"]}, image_dir)
    have_img = {c for p in plans for c in p["cands"]
                if (image_dir / safe_name(*c)).exists()}
    all_c = {c for p in plans for c in p["cands"]}
    print(f"{len(have_img)}/{len(all_c)} candidate images on disk "
          f"(missing ones are skipped in the vision condition)")

    conditions = {"vision_sonnet": (AnthropicJudge(SONNET), True),
                  "text_sonnet": (AnthropicJudge(SONNET), False)}
    if os.environ.get("DEEPSEEK_API_KEY"):
        conditions["text_deepseek"] = (
            DeepSeekJudge(args.deepseek_model), False)
    else:
        print("WARNING: DEEPSEEK_API_KEY unset, skipping text_deepseek")

    cache = Cache(f"{args.out}_cache.jsonl")
    depths = sorted({d for d in [20, args.verify_k] if d <= args.verify_k})
    results = []
    alternatives = []   # endorsed non-target figures, for manual audit

    for n, p in enumerate(plans):
        for cond, (judge, uses_image) in conditions.items():
            verdicts = {}
            for c in p["cands"]:
                key = f"{PROMPT_VER}:{cond}:{p['register']}:" \
                      f"{p['target'][0]}:{p['target'][1]}:{c[0]}:{c[1]}"
                v = cache.get(key)
                if v is None:
                    img = image_dir / safe_name(*c) if uses_image else None
                    if uses_image and not img.exists():
                        continue
                    prompt = VERIFY_PROMPT.format(
                        query=p["query"],
                        materials_line=MATERIALS_VISION if uses_image
                        else MATERIALS_TEXT,
                        caption=captions.get(c, "")[:1500])
                    v = judge.judge(prompt, img_path=img)
                    if v is not None:
                        cache.put(key, v)
                if v is not None:
                    verdicts[c] = v

            for tau in TAUS:
                def accept(c):
                    v = verdicts.get(c)
                    return bool(v and v["match"] and v["confidence"] >= tau)
                for d in depths:
                    cands_d = p["cands"][:d]
                    # verified ranking: accepted first by confidence, then
                    # retrieval order
                    order = sorted(
                        range(len(cands_d)),
                        key=lambda i: (0 if accept(cands_d[i]) else 1,
                                       -(verdicts.get(cands_d[i]) or
                                         {"confidence": 0})["confidence"], i))
                    vrank = None
                    for r, i in enumerate(order, 1):
                        if cands_d[i] == p["target"]:
                            vrank = r
                            break
                    top = cands_d[order[0]] if order else None
                    rr = p["ret_rank"] if (p["ret_rank"] is not None
                                           and p["ret_rank"] <= d) else None
                    # audit dump: target missed but judge endorsed the top
                    # candidate (recorded once, at tau=0.5 and full depth)
                    if (tau == 0.5 and d == depths[-1] and rr is None
                            and top is not None and accept(top)):
                        v = verdicts.get(top) or {}
                        alternatives.append({
                            "register": p["register"], "cond": cond,
                            "query": p["query"],
                            "cand_arxiv": top[0], "cand_fig": top[1],
                            "confidence": v.get("confidence"),
                            "reason": v.get("reason", ""),
                            "caption_snippet":
                                captions.get(top, "")[:300],
                            "image": str(image_dir / safe_name(*top)),
                        })
                    results.append({
                        "register": p["register"], "cond": cond,
                        "tau": tau, "depth": d,
                        "target_retrieved": rr is not None,
                        "ret_rank": rr, "ver_rank": vrank,
                        "top_accepted": accept(top) if top else False,
                        "any_accepted": any(accept(c) for c in cands_d),
                        "real_accepted": accept(p["target"])
                        if p["target"] in cands_d else None,
                    })
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(plans)} queries judged")

    df = pd.DataFrame(results)
    df.to_csv(f"{args.out}_raw.csv", index=False)
    if alternatives:
        pd.DataFrame(alternatives).to_csv(
            f"{args.out}_alternatives.csv", index=False)

    print("\n--- Verification summary ---")
    rows = []
    for (reg, cond, tau, d), g in df.groupby(
            ["register", "cond", "tau", "depth"]):
        got = g[g.target_retrieved]
        miss = g[~g.target_retrieved]
        real = got[got.real_accepted.notna()]
        rows.append({
            "register": reg, "cond": cond, "tau": tau, "depth": d,
            "n": len(g),
            "tgt_in_topd": round(g.target_retrieved.mean(), 3),
            "ret_R@1": round((got.ret_rank == 1).mean(), 3)
            if len(got) else None,
            "ver_R@1": round((got.ver_rank == 1).mean(), 3)
            if len(got) else None,
            "ver_R@5": round((got.ver_rank <= 5).mean(), 3)
            if len(got) else None,
            "real_accept%": round(real.real_accepted.mean(), 3)
            if len(real) else None,
            "alt_accept": round(miss.top_accepted.mean(), 3)
            if len(miss) else None,
            "abstain": round((~miss.any_accepted).mean(), 3)
            if len(miss) else None,
        })
    summary = pd.DataFrame(rows).sort_values(
        ["register", "cond", "tau", "depth"])
    summary.to_csv(f"{args.out}_summary.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.to_string(index=False))
    print(f"\nwrote {args.out}_raw.csv, {args.out}_summary.csv")
    print("real_accept% = judge accepts the true figure when it sees it. "
          "alt_accept = when the designated target was NOT retrieved, the "
          "judge endorsed some other figure: for detailed queries this is "
          "close to a false-positive rate; for vague queries the endorsed "
          "figure may legitimately satisfy the query -- audit the dump in "
          f"{args.out}_alternatives.csv to split those cases.")


if __name__ == "__main__":
    main()
