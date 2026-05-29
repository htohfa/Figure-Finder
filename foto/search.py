import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Optional

import requests

from .parser import parse_json


TRIAGE_PROMPT_BATCH = '''You are evaluating whether papers likely contain a figure a researcher is looking for.

Science topic: "{science_query}"
{plot_type_line}

Below are {n} papers numbered 1 to {n}. For each paper, decide if it likely contains a matching figure.

{papers}

CRITICAL: Respond with a JSON array of EXACTLY {n} objects, one per paper in the same order. No wrapper object, no markdown, no preamble. Output must start with `[` and end with `]`.

Each object must have this shape:
{{"relevant": true|false, "confidence": 0.0-1.0, "reason": "one sentence"}}

Example for 2 papers:
[{{"relevant": true, "confidence": 0.8, "reason": "..."}}, {{"relevant": false, "confidence": 0.3, "reason": "..."}}]'''



TRIAGE_PROMPT_SINGLE = '''You are evaluating whether a paper likely contains the figure a researcher is looking for.

Science topic: "{science_query}"
{plot_type_line}

Paper title: {title}
Abstract: {abstract}

Does this paper likely contain a figure matching the description?
Respond JSON only: {{"relevant": <true/false>, "confidence": <0.0-1.0>, "reason": "<one sentence>"}}'''

TRIAGE_PROMPT_BATCH = '''You are evaluating whether papers likely contain a figure a researcher is looking for.

Science topic: "{science_query}"
{plot_type_line}

Below are {n} papers. For each, decide if it likely contains a matching figure.

{papers}

Respond JSON array only, one object per paper in the same order:
[{{"relevant": <true/false>, "confidence": <0.0-1.0>, "reason": "<one sentence>"}}, ...]'''

EXPANSION_PROMPT = '''A researcher wants papers containing this specific figure: "{query}"

Generate exactly 4 short search queries (3-6 words each) covering DIFFERENT subfields/contexts:
- One observational/survey-focused
- One simulation/theory-focused
- One review/compilation-focused
- One using alternative axis variable names or notation

Respond JSON list only: ["q1", "q2", "q3", "q4"]'''

ADJACENT_PROMPT = '''Highly-cited papers found while searching for: "{query}"

{titles}

Suggest 2 short (3-6 word) search queries for ADJACENT subfields likely to contain
this kind of figure but missed by the original search. JSON list only.'''


class PaperSearcher:
    def __init__(self, s2_key: Optional[str] = None):
        self.s2_key = s2_key

    def search_s2(self, query: str, limit: int = 10) -> list[dict]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query, "limit": limit,
            "fields": "paperId,title,authors,year,abstract,openAccessPdf,externalIds,citationCount",
        }
        headers = {"x-api-key": self.s2_key} if self.s2_key else {}
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=30)
                if r.status_code == 429:
                    time.sleep(2 ** attempt + 2)
                    continue
                r.raise_for_status()
                results = r.json().get("data", [])
                for p in results:
                    p["_source"] = "s2"
                return results
            except Exception:
                time.sleep(2 ** attempt + 2)
        return []

    def search_arxiv(self, query: str, limit: int = 10) -> list[dict]:
        params = {
            "search_query": f"all:{query}",
            "start": 0, "max_results": limit, "sortBy": "relevance",
        }
        url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return self._parse_arxiv_atom(r.text)
        except Exception:
            return []

    def search_arxiv_by_author(self, author: str, limit: int = 8) -> list[dict]:
        params = {
            "search_query": f'au:"{author}"',
            "start": 0, "max_results": limit,
            "sortBy": "submittedDate", "sortOrder": "descending",
        }
        url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            papers = self._parse_arxiv_atom(r.text)
            for p in papers:
                p["_source"] = "arxiv_author"
            return papers
        except Exception:
            return []

    def _parse_arxiv_atom(self, xml_text: str) -> list[dict]:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        papers = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1].split("v")[0]
            papers.append({
                "paperId": f"arxiv_{arxiv_id}",
                "title": entry.find("atom:title", ns).text.strip().replace("\n", " "),
                "abstract": entry.find("atom:summary", ns).text.strip().replace("\n", " "),
                "year": int(entry.find("atom:published", ns).text[:4]),
                "authors": [{"name": a.find("atom:name", ns).text}
                            for a in entry.findall("atom:author", ns)],
                "externalIds": {"ArXiv": arxiv_id},
                "openAccessPdf": {"url": f"https://arxiv.org/pdf/{arxiv_id}"},
                "citationCount": 0,
                "_source": "arxiv",
            })
        return papers

    def _search_one(self, query: str, limit: int = 10) -> list[dict]:
        hits = self.search_s2(query, limit=limit)
        return hits if hits else self.search_arxiv(query, limit=limit)

    def _dedupe(self, papers: list[dict], seen_ids: set, seen_titles: set) -> list[dict]:
        out = []
        for p in papers:
            title_key = p.get("title", "").lower().strip()
            key = p.get("externalIds", {}).get("ArXiv") or p["paperId"]
            if key not in seen_ids and title_key not in seen_titles:
                seen_ids.add(key)
                seen_titles.add(title_key)
                out.append(p)
        return out

    def _top_authors(self, papers: list[dict], top_n: int = 3, min_citations: int = 50) -> list[str]:
        scores: Counter = Counter()
        for paper in papers:
            citations = paper.get("citationCount", 0) or 0
            if citations < min_citations:
                continue
            for author in paper.get("authors", [])[:5]:
                if name := author.get("name", ""):
                    scores[name] += citations
        return [name for name, _ in scores.most_common(top_n)]

    def _landmarks(self, papers: list[dict], top_n: int = 3) -> list[dict]:
        scored = [p for p in papers if p.get("citationCount", 0)]
        scored.sort(key=lambda p: -(p.get("citationCount", 0)))
        return scored[:top_n]

    def expanded_search_pathfinder(self, query: str, openai_key: str, log=None) -> list[dict]:
        from .pathfinder_search import PathfinderSearcher
        searcher = PathfinderSearcher(openai_key=openai_key)
        if log:
            log("  Semantic search via Pathfinder corpus...")
        return searcher.search(query, limit=50)

    def expanded_search(self, query: str, client, model_id: str, prices: dict, tracker, max_tokens: int = 1000, log=None) -> list[dict]:
        def _log(msg):
            if log:
                log(msg)

        seen_ids, seen_titles = set(), set()
        all_results = []

        try:
            response = client.messages.create(
                model=model_id, max_tokens=max_tokens,
                messages=[{"role": "user", "content": EXPANSION_PROMPT.format(query=query)}],
            )
            tracker.record(
                "expansion", model_id, prices,
                response.usage.input_tokens, response.usage.output_tokens,
            )
            extra_queries = parse_json(response.content[0].text)
        except Exception:
            extra_queries = []

        queries = [query] + [q for q in extra_queries if q != query]
        _log(f"  Searching {len(queries)} query variants...")

        for q in queries:
            hits = self._search_one(q, limit=10)
            new = self._dedupe(hits, seen_ids, seen_titles)
            all_results.extend(new)
            time.sleep(0.5)

        _log(f"  {len(all_results)} papers after round 1")

        top_authors = self._top_authors(all_results)
        if top_authors:
            for author in top_authors:
                hits = self.search_arxiv_by_author(author, limit=8)
                new = self._dedupe(hits, seen_ids, seen_titles)
                all_results.extend(new)
                time.sleep(1)
            _log(f"  {len(all_results)} papers after author search")

        landmarks = self._landmarks(all_results)
        if landmarks:
            titles = "\n".join(f"- {p['title']}" for p in landmarks)
            try:
                response = client.messages.create(
                    model=model_id, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": ADJACENT_PROMPT.format(query=query, titles=titles)}],
                )
                tracker.record(
                    "expansion", model_id, prices,
                    response.usage.input_tokens, response.usage.output_tokens,
                )
                adjacent_queries = parse_json(response.content[0].text)
                for q in adjacent_queries:
                    hits = self._search_one(q, limit=8)
                    new = self._dedupe(hits, seen_ids, seen_titles)
                    all_results.extend(new)
                    time.sleep(0.5)
                _log(f"  {len(all_results)} papers after adjacent topic search")
            except Exception:
                pass

        all_results.sort(key=lambda p: -(p.get("citationCount", 0) or 0))
        return all_results


class PaperTriager:
    def __init__(self, client, model_id: str, prices: dict, tracker,
                 max_tokens: int = 200, batch_size: int = 1):
        self.client = client
        self.model_id = model_id
        self.prices = prices
        self.tracker = tracker
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    def triage(self, papers: list[dict], spec: dict) -> list[dict]:
        if self.batch_size <= 1:
            return self._triage_single(papers, spec)
        return self._triage_batched(papers, spec)

    def _triage_single(self, papers: list[dict], spec: dict) -> list[dict]:
        plot_line = f"Plot type: {spec['plot_type']}" if spec.get("plot_type") else ""
        scored = []
        for paper in papers:
            prompt = TRIAGE_PROMPT_SINGLE.format(
                science_query=spec.get("science_query") or "(unspecified)",
                plot_type_line=plot_line,
                title=paper.get("title", ""),
                abstract=(paper.get("abstract") or "")[:800],
            )
            try:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.tracker.record(
                    "triage", self.model_id, self.prices,
                    response.usage.input_tokens, response.usage.output_tokens,
                )
                result = parse_json(response.content[0].text)
                if isinstance(result, list) and result:
                    result = result[0]
                paper["_triage"] = result
                if result.get("relevant"):
                    scored.append(paper)
            except Exception:
                paper["_triage"] = {"relevant": False, "confidence": 0}

        scored.sort(key=lambda p: -(p.get("citationCount") or 0) * 0.1
                                  - p["_triage"].get("confidence", 0))
        return scored

    def _triage_batched(self, papers: list[dict], spec: dict) -> list[dict]:
        plot_line = f"Plot type: {spec['plot_type']}" if spec.get("plot_type") else ""
        scored = []
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i:i + self.batch_size]
            batch_text = "\n\n".join(
                f"Paper {j+1}:\nTitle: {p.get('title','')}\nAbstract: {(p.get('abstract') or '')[:600]}"
                for j, p in enumerate(batch)
            )
            prompt = TRIAGE_PROMPT_BATCH.format(
                science_query=spec.get("science_query") or "(unspecified)",
                plot_type_line=plot_line,
                n=len(batch),
                papers=batch_text,
            )
            try:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.tracker.record(
                    "triage_batch", self.model_id, self.prices,
                    response.usage.input_tokens, response.usage.output_tokens,
                )
                results = parse_json(response.content[0].text)
                if isinstance(results, dict):
                    results = [results]
                for paper, result in zip(batch, results):
                    if isinstance(result, list) and result:
                        result = result[0]
                    paper["_triage"] = result
                    if result.get("relevant"):
                        scored.append(paper)
            except Exception:
                for paper in batch:
                    paper["_triage"] = {"relevant": False, "confidence": 0}

        scored.sort(key=lambda p: -(p.get("citationCount") or 0) * 0.1
                                  - p["_triage"].get("confidence", 0))
        return scored
