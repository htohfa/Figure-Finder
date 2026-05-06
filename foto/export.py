import io
import json
import zipfile


def format_authors(paper: dict, max_authors: int = 3) -> str:
    authors = paper.get("authors", [])
    names = ", ".join(a["name"] for a in authors[:max_authors])
    if len(authors) > max_authors:
        names += " et al."
    return names


def get_confidence(match: dict) -> float:
    verdict = match.get("_verify") or match.get("_primary") or {}
    return verdict.get("confidence", 0)


def confidence_badge_class(conf: float) -> str:
    if conf >= 0.8:
        return "badge-high"
    if conf >= 0.6:
        return "badge-mid"
    return "badge-low"


def build_zip(matches: list[dict], paper_lookup: dict) -> bytes:
    buf = io.BytesIO()
    summary = []

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, match in enumerate(matches):
            paper = paper_lookup.get(match["paper_id"], {})
            verdict = match.get("_verify") or match.get("_primary") or {}
            fname = f"figure_{i+1:02d}_p{match['page']}_{match['paper_id'][:12]}.png"
            zf.writestr(fname, match["image_bytes"])
            summary.append({
                "filename": fname,
                "paper_title": paper.get("title", ""),
                "authors": format_authors(paper),
                "year": paper.get("year"),
                "arxiv_id": paper.get("externalIds", {}).get("ArXiv"),
                "page": match["page"],
                "confidence": verdict.get("confidence", 0),
                "plot_type": verdict.get("plot_type", ""),
                "what_is_plotted": verdict.get("what_is_plotted", ""),
                "caption": (match.get("caption") or "")[:300],
            })
        zf.writestr("summary.json", json.dumps(summary, indent=2))

    buf.seek(0)
    return buf.read()
