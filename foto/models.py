from dataclasses import dataclass


@dataclass
class ModelConfig:
    label: str
    model_id: str
    provider: str
    api_help_url: str
    api_help_text: str
    prices: dict
    free_tier: bool = False
    triage_max_tokens: int = 200
    score_max_tokens: int = 600
    verify_max_tokens: int = 800
    triage_batch_size: int = 1
    score_batch_size: int = 1
    estimated_time: str = ""


MODEL_REGISTRY = [
    ModelConfig(
        label="Claude Sonnet 4.5  (paid · ~2 min)",
        model_id="claude-sonnet-4-5-20250929",
        provider="anthropic",
        api_help_url="https://console.anthropic.com/settings/keys",
        api_help_text="Go to this website and get your API key there.",
        prices={"input": 3.00, "output": 15.00},
        triage_max_tokens=150,
        score_max_tokens=400,
        verify_max_tokens=500,
        triage_batch_size=5,
        score_batch_size=5,
        estimated_time="~ 3 min",
    ),
    ModelConfig(
        label="Claude Haiku 4.5  (paid · ~1 min)",
        model_id="claude-haiku-4-5-20251001",
        provider="anthropic",
        api_help_url="https://console.anthropic.com/settings/keys",
        api_help_text="Go to this website and get your API key there.",
        prices={"input": 1.00, "output": 5.00},
        triage_max_tokens=150,
        score_max_tokens=400,
        verify_max_tokens=500,
        triage_batch_size=5,
        score_batch_size=5,
        estimated_time="~3 min",
    ),
    ModelConfig(
        label="Gemini 2.5 Flash-Lite  (free · ~4 min)",
        model_id="gemini-2.5-flash-lite",
        provider="gemini",
        api_help_url="https://ai.google.dev/gemini-api/docs/api-key",
        api_help_text="Go to this website and get your API key for free.",
        prices={"input": 0.0, "output": 0.0},
        free_tier=True,
        triage_max_tokens=2000,
        score_max_tokens=4000,
        verify_max_tokens=3000,
        triage_batch_size=5,
        score_batch_size=1,
        estimated_time="~6 min",
    ),
    ModelConfig(
        label="Gemini 2.5 Flash  (free · ~5 min)",
        model_id="gemini-2.5-flash",
        provider="gemini",
        api_help_url="https://ai.google.dev/gemini-api/docs/api-key",
        api_help_text="Go to this website and get your API key for free.",
        prices={"input": 0.0, "output": 0.0},
        free_tier=True,
        triage_max_tokens=2000,
        score_max_tokens=4000,
        verify_max_tokens=3000,
        triage_batch_size=5,
        score_batch_size=1,
        estimated_time="~5 min",
    ),
    ModelConfig(
        label="DeepSeek V3  (5M token trial · then paid · text-only)",
        model_id="deepseek-chat",
        provider="deepseek",
        api_help_url="https://platform.deepseek.com/api_keys",
        api_help_text="Go to this website and get your API key. New accounts get 5M free tokens.",
        prices={"input": 0.27, "output": 1.10},
        triage_max_tokens=300,
        score_max_tokens=600,
        verify_max_tokens=800,
        triage_batch_size=1,
        score_batch_size=1,
        estimated_time="~3 min",
    ),
]

MODEL_LABELS = [m.label for m in MODEL_REGISTRY]


def get_model(label: str) -> ModelConfig:
    for m in MODEL_REGISTRY:
        if m.label == label:
            return m
    raise ValueError(f"Unknown model label: {label}")


PROVIDER_DISPLAY = {
    "anthropic": "Anthropic API Key",
    "gemini": "Google Gemini API Key",
    "deepseek": "DeepSeek API Key",
}


class CostTracker:
    def __init__(self):
        self.log = []

    def record(self, stage: str, model_id: str, prices: dict, input_tokens: int, output_tokens: int):
        self.log.append({
            "stage": stage,
            "model": model_id,
            "prices": prices,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

    def total(self) -> float:
        cost = 0.0
        for entry in self.log:
            p = entry["prices"]
            cost += (entry["input_tokens"] / 1e6) * p.get("input", 0)
            cost += (entry["output_tokens"] / 1e6) * p.get("output", 0)
        return cost
