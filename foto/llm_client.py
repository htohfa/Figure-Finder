from dataclasses import dataclass
from threading import Lock
import base64
import time


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class Response:
    content: list
    usage: Usage
    model: str


_RATE_LIMITS = {
    "anthropic": 0.0,
    "gemini": 5.0,   # ~12 RPM, safe under 15 RPM free-tier ceiling
    "deepseek": 0.0,
}
_last_call_time: dict = {}
_lock = Lock()


def _pace(provider: str):
    delay = _RATE_LIMITS.get(provider, 0.0)
    if delay <= 0:
        return
    with _lock:
        last = _last_call_time.get(provider, 0.0)
        wait = delay - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        _last_call_time[provider] = time.time()


class LLMClient:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self.messages = _MessagesNamespace(self)

    def _call(self, model: str, max_tokens: int, messages: list) -> Response:
        _pace(self.provider)
        if self.provider == "anthropic":
            return self._call_anthropic(model, max_tokens, messages)
        if self.provider == "gemini":
            return self._call_gemini(model, max_tokens, messages)
        if self.provider == "deepseek":
            return self._call_deepseek(model, max_tokens, messages)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _call_anthropic(self, model, max_tokens, messages):
        from anthropic import Anthropic
        client = Anthropic(api_key=self.api_key)
        resp = client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
        return Response(
            content=[TextBlock(text=b.text) for b in resp.content if hasattr(b, "text")],
            usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            model=model,
        )

    def _call_gemini(self, model, max_tokens, messages):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.api_key)

        contents = []
        for msg in messages:
            parts = []
            content = msg["content"]
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            else:
                for block in content:
                    if block.get("type") == "text":
                        parts.append(types.Part.from_text(text=block["text"]))
                    elif block.get("type") == "image":
                        src = block["source"]
                        img_bytes = base64.b64decode(src["data"]) if src["type"] == "base64" else src["data"]
                        parts.append(types.Part.from_bytes(
                            data=img_bytes, mime_type=src.get("media_type", "image/png"),
                        ))
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=parts))

        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = resp.text or ""
        usage = resp.usage_metadata
        return Response(
            content=[TextBlock(text=text)],
            usage=Usage(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            ),
            model=model,
        )

    def _call_deepseek(self, model, max_tokens, messages):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
        oai_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                oai_messages.append({"role": msg["role"], "content": content})
            else:
                text_parts = [b["text"] for b in content if b.get("type") == "text"]
                oai_messages.append({"role": msg["role"], "content": "\n".join(text_parts)})

        resp = None
        last_err = None
        for attempt in range(4):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                    time.sleep(2 ** attempt * 5)  # 5, 10, 20, 40 sec backoff
                    continue
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(60)
                    continue
                raise
        if resp is None:
            raise last_err


class _MessagesNamespace:
    def __init__(self, parent):
        self._parent = parent

    def create(self, model: str, max_tokens: int, messages: list, **kwargs) -> Response:
        return self._parent._call(model, max_tokens, messages)
