import json
from pathlib import Path
from datetime import datetime, timezone


STATS_PATH = Path("/data/foto-stats.json")
FALLBACK_PATH = Path.home() / ".cache" / "foto" / "foto-stats.json"


def _stats_file() -> Path:
    if STATS_PATH.parent.exists():
        return STATS_PATH
    FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    return FALLBACK_PATH


def _read() -> dict:
    path = _stats_file()
    if not path.exists():
        return {"searches": 0, "ratings": [], "log": []}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"searches": 0, "ratings": [], "log": []}


def _write(data: dict):
    try:
        _stats_file().write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def load_stats() -> dict:
    data = _read()
    return {
        "searches": data.get("searches", 0),
        "ratings": data.get("ratings", []),
    }


def log_search(query: str, n_papers: int, n_matches: int, cost: float):
    data = _read()
    data["searches"] = data.get("searches", 0) + 1
    log_entries = data.setdefault("log", [])
    log_entries.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "n_papers": n_papers,
        "n_matches": n_matches,
        "cost": round(cost, 4),
        "rating": None,
    })
    _write(data)


def log_rating(rating: int):
    data = _read()
    data.setdefault("ratings", []).append(rating)
    log_entries = data.get("log", [])
    for entry in reversed(log_entries):
        if entry.get("rating") is None:
            entry["rating"] = rating
            break
    _write(data)
