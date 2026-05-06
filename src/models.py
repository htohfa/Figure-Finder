from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    label: str
    smart: str  # used for parsing and verification
    cheap: str  # used for triage and primary scoring
    prices: dict = field(default_factory=dict)


# Add new models here - that's the only thing you need to touch
MODEL_REGISTRY = [
    ModelConfig(
        label="Haiku 4.5  (fast · cheap)",
        smart="claude-haiku-4-5-20251001",
        cheap="claude-haiku-4-5-20251001",
        prices={
            "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        },
    ),
    ModelConfig(
        label="Sonnet 4.5  (balanced)",
        smart="claude-sonnet-4-5-20250929",
        cheap="claude-haiku-4-5-20251001",
        prices={
            "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
            "claude-haiku-4-5-20251001":  {"input": 1.00, "output": 5.00},
        },
    ),
    ModelConfig(
        label="Sonnet 4.5  (both smart)",
        smart="claude-sonnet-4-5-20250929",
        cheap="claude-sonnet-4-5-20250929",
        prices={
            "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        },
    ),
]

MODEL_LABELS = [m.label for m in MODEL_REGISTRY]


def get_model(label: str) -> ModelConfig:
    for m in MODEL_REGISTRY:
        if m.label == label:
            return m
    raise ValueError(f"Unknown model label: {label}")


class CostTracker:
    def __init__(self, prices: dict):
        self.prices = prices
        self.log = []

    def record(self, stage: str, model: str, response):
        self.log.append({
            "stage": stage,
            "model": model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

    def total(self) -> float:
        cost = 0.0
        for entry in self.log:
            p = self.prices.get(entry["model"], {"input": 0, "output": 0})
            cost += (entry["input_tokens"] / 1e6) * p["input"]
            cost += (entry["output_tokens"] / 1e6) * p["output"]
        return cost
