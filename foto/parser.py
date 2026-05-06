import json
import base64
import io
from typing import Optional

from PIL import Image as PILImage


PARSE_TEXT_PROMPT = '''A researcher is looking for a scientific figure. Their description:

"{query}"

Extract these fields:
1. science_query — what physics/topic to search for, in 3-8 words. Strip plot-type words and keep only the science.
2. plot_type — if explicitly mentioned, one of: scatter, line, scatter+line, contour, banana_contour, corner_plot, heatmap, histogram, error_bar, image_with_overlay, multi_panel, sky_map, power_spectrum, other. If NOT mentioned, return null.
3. axis_x, axis_y — if specific axes are mentioned, extract them; otherwise null.

Respond JSON only:
{{"science_query": "<...>", "plot_type": <"..." or null>, "axis_x": <"..." or null>, "axis_y": <"..." or null>}}'''

PARSE_SKETCH_PROMPT = '''Look at this hand-drawn sketch of a scientific plot.

Identify:
1. plot_type — one of: scatter, line, scatter+line, contour, banana_contour, corner_plot, heatmap, histogram, error_bar, image_with_overlay, multi_panel, sky_map, power_spectrum, other. Use null if unclear.
2. axis_x, axis_y — read any labels visible on the axes. If unlabeled, return null. Don't guess.
3. science_query — IF axis labels imply a recognizable scientific topic, suggest a 3-8 word search query. If you can't tell, return null.

Respond JSON only:
{{"plot_type": <"..." or null>, "axis_x": <"..." or null>, "axis_y": <"..." or null>, "science_query": <"..." or null>}}'''


def encode_image(data: bytes, max_dim: int = 768) -> str:
    image = PILImage.open(io.BytesIO(data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), PILImage.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def parse_json(text: str) -> dict:
    return json.loads(text.strip().replace("```json", "").replace("```", "").strip())


class InputParser:
    def __init__(self, client, model: str, tracker):
        self.client = client
        self.model = model
        self.tracker = tracker

    def _call(self, messages: list) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=messages,
        )
        self.tracker.record("parse_input", self.model, response)
        return parse_json(response.content[0].text)

    def _from_text(self, text: str) -> dict:
        return self._call([{
            "role": "user",
            "content": PARSE_TEXT_PROMPT.format(query=text),
        }])

    def _from_sketch(self, sketch_bytes: bytes) -> dict:
        return self._call([{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": encode_image(sketch_bytes),
                }},
                {"type": "text", "text": PARSE_SKETCH_PROMPT},
            ],
        }])

    def parse(self, text: Optional[str] = None, sketch_bytes: Optional[bytes] = None) -> dict:
        if not text and not sketch_bytes:
            raise ValueError("Need at least text or a sketch.")

        spec = {
            "science_query": None, "plot_type": None,
            "axis_x": None, "axis_y": None,
            "has_sketch": sketch_bytes is not None,
            "sketch_bytes": sketch_bytes,
        }

        if text and sketch_bytes:
            t = self._from_text(text)
            s = self._from_sketch(sketch_bytes)
            spec["science_query"] = t.get("science_query") or s.get("science_query")
            spec["plot_type"] = t.get("plot_type") or s.get("plot_type")
            spec["axis_x"] = t.get("axis_x") or s.get("axis_x")
            spec["axis_y"] = t.get("axis_y") or s.get("axis_y")
        elif text:
            spec.update(self._from_text(text))
        else:
            spec.update(self._from_sketch(sketch_bytes))

        spec["has_sketch"] = sketch_bytes is not None
        spec["sketch_bytes"] = sketch_bytes
        return spec
