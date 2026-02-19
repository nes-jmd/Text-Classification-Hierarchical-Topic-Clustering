from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class BaseLabeler(ABC):
    @abstractmethod
    def label(self, snippets: list[str]) -> dict:
        raise NotImplementedError


class HeuristicLabeler(BaseLabeler):
    def label(self, snippets: list[str]) -> dict:
        tokens = []
        for snippet in snippets:
            words = re.findall(r"[A-Za-z]{3,}", snippet.lower())
            tokens.extend([w for w in words if w not in ENGLISH_STOP_WORDS])
        common = [w for w, _ in Counter(tokens).most_common(4)]
        label = " ".join(common[:3]).title() if common else "General Topic"
        rationale = (
            f"This topic is represented by recurring terms: {', '.join(common[:4])}."
            if common
            else "The snippets are broad, so this is a general topic."
        )
        return {"label": label[:40], "rationale": rationale}


class OpenAILabeler(BaseLabeler):
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def label(self, snippets: list[str]) -> dict:
        prompt = {
            "task": "Label a text cluster.",
            "requirements": {
                "output_json": {"label": "2-6 words", "rationale": "1 sentence"},
                "json_only": True,
            },
            "snippets": snippets,
        }
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You generate concise topic labels. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            response_format={"type": "json_object"},
        )
        text = (response.choices[0].message.content or "").strip()
        return _safe_json_payload(text)


def _safe_json_payload(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}

    label = str(payload.get("label", "General Topic")).strip()
    rationale = str(payload.get("rationale", "Label generated from representative snippets.")).strip()
    return {"label": label[:48], "rationale": rationale}


def get_labeler() -> BaseLabeler:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAILabeler()
    print("[WARN] OPENAI_API_KEY not found. Using heuristic labeler.")
    return HeuristicLabeler()
