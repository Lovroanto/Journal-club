from __future__ import annotations

from dataclasses import dataclass
import json
import requests
from typing import Any, Dict


@dataclass
class OllamaClient:
    """
    Minimal Ollama REST client.
    Assumes Ollama is already running; does NOT attempt to start/stop it.
    """
    base_url: str = "http://localhost:11434"
    timeout_s: int = 120

    def generate_json(self, *, model: str, system: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls Ollama /api/generate using structured outputs via JSON schema in `format`.
        Returns parsed JSON dict.

        Note: Ollama commonly returns {"response": "<json-string>", ...}.
        """
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "format": schema,
            "stream": False,
        }

        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()

        resp = (data.get("response") or "").strip()
        if not resp:
            return {}

        # Strict parse, then lenient fallback
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            start = resp.find("{")
            end = resp.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(resp[start:end + 1])
            raise
