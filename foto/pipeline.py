import time
from typing import Optional

import fitz
import requests
import streamlit as st

from .parser import encode_image, parse_json


PRIMARY_PROMPT = '''A researcher is looking for a scientific figure.

Science topic: "{science_query}"
{plot_type_line}
{axis_lines}

Look at this figure and read its caption. Score how well it matches.

Caption: {caption}

SCORING GUIDANCE:
Science match is the dominant factor. A figure on the wrong scientific topic should never score above 0.4.
If a plot type is specified, treat similarity as a gradient:
  Strong match: scatter ↔ scatter+line ↔ error_bar; contour ↔ banana_contour ↔ corner_plot; heatmap ↔ image_with_overlay
  Partial match (~0.5 weight): scatter ↔ histogram; line ↔ power_spectrum; any specific type ↔ multi_panel
  Weak/no match: scatter/line vs contour; histogram vs corner_plot
If specific axes are specified: semantically equivalent labels match (redshift ≈ z; Omega_m ≈ Ω_m).
{sketch_line}

Respond JSON only:
{{
  "matches_request": <true/false>,
  "confidence": <0-1>,
  "plot_type": "<observed plot type>",
  "science_match": <true/false>,
  "plot_type_match": <"strong"|"partial"|"weak"|"not_specified">,
  "what_is_plotted": "<one sentence>",
  "reason": "<one sentence>"
}}'''

VERIFY_PROMPT = '''A researcher is looking for a scientific figure.

Science topic: "{science_query}"
{plot_type_line}
{axis_lines}

You are doing a careful second-pass verification. This figure already passed a first-pass filter.
Caption: {caption}
{sketch_line}

Look very carefully and respond JSON only:
{{
  "matches_request": <true/false>,
  "confidence": <0-1>,
  "x_axis_text": "<what is actually on the x-axis, verbatim>",
  "y_axis_text": "<what is actually on the y-axis, verbatim>",
  "plot_type": "<plot type>",
  "plot_type_match": <"strong"|"partial"|"weak"|"not_specified">,
  "what_is_plotted": "<one sentence>",
  "reason": "<two sentences>"
}}'''

CAPTION_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "with", "for", "in", "on", "at",
    "to", "from", "vs", "versus", "as", "is", "are", "by",
    "plot", "figure", "showing", "shows", "graph", "chart",
}


class PDFStore:
    """Holds PDFs in session memory so the same paper isn't re-fetched."""

    @staticmethod
    def get(paper_id: str) -> Optional[bytes]:
        return st.session_state.pdf_cache.get(paper_id)

    @staticmethod
    def put(paper_id: str, data: bytes):
        st.session_state.pdf_cache[paper_id] = data

    @staticmethod
    def fetch(paper: dict) -> tuple[Optional[bytes], str]:
        paper_id = paper["paperId"]
        cached = PDFStore.get(paper_id)
        if cached:
            return cached, "cached"

        url = None
        if arxiv_id := paper.get("externalIds", {}).get("ArXiv"):
            url = f"https://arxiv.org/pdf/{arxiv_id}"
        elif oa := paper.get("openAccessPdf"):
            url = oa.get("url")

        if not url:
            return None, "no PDF URL"

        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "research-agent/0.1"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                return None, "not a PDF"
            PDFStore.put(paper_id, r.content)
            return r.content, "downloaded"
        except requests.exceptions.Timeout:
            return None, "timeout"
        except Exception as e:
            return None, str(e)[:60]


class FigureExtractor:
    def extract(self, pdf_bytes: bytes, paper_id: str) -> list[dict]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        figures = []

        for page_num, page in enumerate(doc):
            captions = self._find_captions(page)
            for image in page.get_images(full=True):
                xref = image[0]
                try:
                    fig = self._extract_image(doc, page, xref, page_num, paper_id, captions)
                    if fig:
                        figures.append(fig)
                except Exception:
                    continue

        if len(figures) < 2:
            figures.extend(self._fallback_page_render(doc, paper_id))

        doc.close()
        return figures

    def _find_captions(self, page) -> list[dict]:
        captions = []
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if text.lower().startswith(("figure ", "fig. ", "fig ")):
                captions.append({"bbox": block[:4], "text": text[:1500]})
        return captions

    def _extract_image(self, doc, page, xref, page_num, paper_id, captions) -> Optional[dict]:
        pixmap = fitz.Pixmap(doc, xref)
        if pixmap.width < 200 or pixmap.height < 200:
            return None
        if pixmap.n - pixmap.alpha > 3:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)

        img_bytes = pixmap.tobytes("png")
        caption = self._match_caption(page, xref, captions)

        return {
            "paper_id": paper_id,
            "page": page_num + 1,
            "image_bytes": img_bytes,
            "caption": caption,
            "fig_id": f"{paper_id}_p{page_num+1}_x{xref}",
        }

    def _match_caption(self, page, xref, captions) -> str:
        rects = page.get_image_rects(xref)
        if not rects or not captions:
            return ""
        image_bottom = rects[0].y1
        below = [c for c in captions if c["bbox"][1] >= image_bottom - 30]
        if not below:
            return ""
        return min(below, key=lambda c: c["bbox"][1] - image_bottom)["text"]

    def _fallback_page_render(self, doc, paper_id) -> list[dict]:
        figures = []
        for page_num, page in enumerate(doc):
            caption_blocks = [b for b in page.get_text("blocks")
                              if b[4].strip().lower().startswith(("figure ", "fig. "))]
            if not caption_blocks:
                continue
            pixmap = page.get_pixmap(dpi=150)
            figures.append({
                "paper_id": paper_id,
                "page": page_num + 1,
                "image_bytes": pixmap.tobytes("png"),
                "caption": caption_blocks[0][4][:1500],
                "fig_id": f"{paper_id}_p{page_num+1}_full",
            })
        return figures

    def caption_filter(self, figures: list[dict], query: str) -> list[dict]:
        query_words = [
            w.lower().strip(".,;:") for w in query.split()
            if len(w) >= 4 and w.lower() not in CAPTION_STOPWORDS
        ]
        if not query_words:
            return figures
        return [
            fig for fig in figures
            if not fig.get("caption")
            or any(w in fig["caption"].lower() for w in query_words)
        ]


class FigureScorer:
    def __init__(self, client, model: str, tracker):
        self.client = client
        self.model = model
        self.tracker = tracker

    def _build_content(self, figure: dict, spec: dict, prompt_text: str) -> list:
        content = []
        if spec.get("has_sketch") and spec.get("sketch_bytes"):
            content.append({"type": "text", "text": "Researcher's sketch:"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": encode_image(spec["sketch_bytes"]),
            }})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": encode_image(figure["image_bytes"]),
        }})
        content.append({"type": "text", "text": prompt_text})
        return content

    def _format_prompt(self, template: str, figure: dict, spec: dict) -> str:
        plot_type_line = f"Required plot type: {spec['plot_type']}" if spec.get("plot_type") else ""
        axis_lines = ""
        if spec.get("axis_x"):
            axis_lines += f"Required x-axis: {spec['axis_x']}\n"
        if spec.get("axis_y"):
            axis_lines += f"Required y-axis: {spec['axis_y']}"
        sketch_line = "- Structural similarity to the provided sketch" if spec.get("has_sketch") else ""
        return template.format(
            science_query=spec.get("science_query") or "(not specified)",
            plot_type_line=plot_type_line,
            axis_lines=axis_lines,
            sketch_line=sketch_line,
            caption=figure.get("caption") or "(no caption)",
        )

    def score(self, figure: dict, spec: dict) -> dict:
        prompt = self._format_prompt(PRIMARY_PROMPT, figure, spec)
        content = self._build_content(figure, spec, prompt)
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=400,
                messages=[{"role": "user", "content": content}],
            )
            self.tracker.record("score_primary", self.model, response)
            result = parse_json(response.content[0].text)
            figure["_primary"] = result
            return result
        except Exception:
            figure["_primary"] = {"matches_request": False, "confidence": 0}
            return figure["_primary"]

    def verify(self, figure: dict, spec: dict) -> dict:
        prompt = self._format_prompt(VERIFY_PROMPT, figure, spec)
        content = self._build_content(figure, spec, prompt)
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=500,
                messages=[{"role": "user", "content": content}],
            )
            self.tracker.record("verify", self.model, response)
            result = parse_json(response.content[0].text)
            figure["_verify"] = result
            return result
        except Exception:
            figure["_verify"] = {"matches_request": False, "confidence": 0}
            return figure["_verify"]
