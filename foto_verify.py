"""Verification judges: Claude / OpenAI (vision) and DeepSeek (text-only),
sharing the specificity-scaled prompt from the paper benchmark."""
import base64
import io
import json
import time

MODELS = {
    "Claude Sonnet 4.6": dict(provider="anthropic", model="claude-sonnet-4-6",
                              vision=True, price=(3.0, 15.0),
                              key_url="https://console.anthropic.com/settings/keys",
                              key_label="Anthropic API Key"),
    "Claude Haiku 4.5": dict(provider="anthropic",
                             model="claude-haiku-4-5-20251001",
                             vision=True, price=(1.0, 5.0),
                             key_url="https://console.anthropic.com/settings/keys",
                             key_label="Anthropic API Key"),
    "OpenAI GPT-4o": dict(provider="openai", model="gpt-4o",
                          vision=True, price=(2.5, 10.0),
                          key_url="https://platform.openai.com/api-keys",
                          key_label="OpenAI API Key"),
    "DeepSeek (text-only)": dict(provider="deepseek", model="deepseek-chat",
                                 vision=False, price=(0.27, 1.1),
                                 key_url="https://platform.deepseek.com/api_keys",
                                 key_label="DeepSeek API Key"),
}

PROMPT = """A researcher is searching for a published scientific figure with \
this query:

"{query}"

{materials_line}

Caption: {caption}

Judge the candidate against what the query actually specifies. If the query \
is broad or vague, any figure genuinely showing what it describes counts as \
a match; do not demand details the query never mentions. If the query \
specifies particulars (quantities, axes, plot type, what is compared), the \
figure must show them. Science match is the dominant factor: a figure on \
the wrong scientific topic should never score above 0.4.{sketch_line}

Respond JSON only:
{{"match": true|false, "confidence": 0.0-1.0, \
"what_is_plotted": "<one sentence>", "reason": "<one sentence>"}}"""

M_VISION = "Below is the candidate figure image together with its caption."
M_TEXT = "Only the figure's caption is available (no image)."
M_SKETCH = ("\nThe researcher also provided a rough sketch of the figure "
            "they remember (first image); weigh structural similarity to it.")


def _jpeg_b64(path_or_bytes, max_dim=1024):
    from PIL import Image
    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    else:
        im = Image.open(path_or_bytes)
    im = im.convert("RGB")
    im.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _parse(text):
    obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
    return {"match": bool(obj.get("match")),
            "confidence": float(obj.get("confidence", 0.0)),
            "what_is_plotted": str(obj.get("what_is_plotted", ""))[:300],
            "reason": str(obj.get("reason", ""))[:300]}


class Judge:
    def __init__(self, model_label, api_key):
        cfg = MODELS[model_label]
        self.cfg = cfg
        self.calls = 0
        if cfg["provider"] == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        else:
            from openai import OpenAI
            base = "https://api.deepseek.com" \
                if cfg["provider"] == "deepseek" else None
            self.client = OpenAI(api_key=api_key, base_url=base)

    def est_cost(self):
        i, o = self.cfg["price"]
        per = (1300 if self.cfg["vision"] else 500) / 1e6 * i + 80 / 1e6 * o
        return self.calls * per

    def judge(self, query, caption, image_path=None, sketch_bytes=None,
              retries=4):
        vision = self.cfg["vision"] and image_path is not None
        prompt = PROMPT.format(
            query=query,
            materials_line=M_VISION if vision else M_TEXT,
            caption=(caption or "")[:1500],
            sketch_line=M_SKETCH if (sketch_bytes and vision) else "")
        for attempt in range(retries):
            try:
                self.calls += 1
                if self.cfg["provider"] == "anthropic":
                    content = []
                    if sketch_bytes and vision:
                        content.append({"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg",
                            "data": _jpeg_b64(sketch_bytes)}})
                    if vision:
                        content.append({"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg",
                            "data": _jpeg_b64(image_path)}})
                    content.append({"type": "text", "text": prompt})
                    resp = self.client.messages.create(
                        model=self.cfg["model"], max_tokens=300,
                        messages=[{"role": "user", "content": content}])
                    return _parse(resp.content[0].text)
                else:
                    if vision:
                        content = []
                        if sketch_bytes:
                            content.append({"type": "image_url", "image_url": {
                                "url": "data:image/jpeg;base64,"
                                       + _jpeg_b64(sketch_bytes)}})
                        content.append({"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64,"
                                   + _jpeg_b64(image_path)}})
                        content.append({"type": "text", "text": prompt})
                    else:
                        content = prompt
                    resp = self.client.chat.completions.create(
                        model=self.cfg["model"], max_tokens=300,
                        messages=[{"role": "user", "content": content}])
                    return _parse(resp.choices[0].message.content)
            except Exception:
                if attempt == retries - 1:
                    return None
                time.sleep(4 * (attempt + 1))
