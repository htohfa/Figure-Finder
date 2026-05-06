import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from .parser import parse_json


TRIAGE_PROMPT = '''You are evaluating whether a paper likely contains the figure a researcher is looking for.

Science topic: "{science_query}"
{plot_type_line}

Paper title: {title}
Abstract: {abstract}

Does this paper likely contain a figure matching the description?
Respond JSON only: {{"relevant": <true/false>, "confidence": <0.0-1.0>, "reason": "<one sentence>"}}'''


class PaperSearcher:
    def __init__(self, s2_key: Optional[str] = None):
        self.s2_key = s2_key

    def search_s2(self, query: str, limit: int = 15) -> list[dict]:
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

    def search_arxiv(self, query: str, limit: int = 15) -> list[dict]:
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

    def combined_search(self, query: str, limit: int = 15) -> list[dict]:
        s2 = self.search_s2(query, limit=limit)
        arxiv = self.search_arxiv(query, limit=limit // 2)

        seen, papers = set(), []
        for p in s2 + arxiv:
            key = p.get("externalIds", {}).get("ArXiv") or p["paperId"]
            if key not in seen:
                seen.add(key)
                papers.append(p)
        return papers


class PaperTriager:
    def __init__(self, client, model: str, tracker):
        self.client = client
        self.model = model
        self.tracker = tracker

    def triage(self, papers: list[dict], spec: dict) -> list[dict]:
        plot_line = f"Plot type: {spec['plot_type']}" if spec.get("plot_type") else ""
        scored = []
        for paper in papers:
            prompt = TRIAGE_PROMPT.format(
                science_query=spec.get("science_query") or "(unspecified)",
                plot_type_line=plot_line,
                title=paper.get("title", ""),
                abstract=(paper.get("abstract") or "")[:800],
            )
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.tracker.record("triage", self.model, response)
                result = parse_json(response.content[0].text)
                paper["_triage"] = result
                if result.get("relevant"):
                    scored.append(paper)
            except Exception:
                paper["_triage"] = {"relevant": False, "confidence": 0}

        scored.sort(key=lambda p: -(p.get("citationCount") or 0) * 0.1
                                  - p["_triage"].get("confidence", 0))
        return scored
