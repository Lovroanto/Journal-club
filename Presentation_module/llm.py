# Presentation_module/llm.py

import subprocess
from typing import Optional


def call_llm(model_name: str, prompt: str, temperature: Optional[float] = None) -> str:
    """
    Call a local LLM via Ollama CLI.

    Args:
        model_name: e.g. "llama3:70b"
        prompt: complete prompt string
        temperature: optional sampling temperature (if supported by your model)

    Returns:
        Raw text output from the model (stripped).
    """
    cmd = ["ollama", "run", model_name]
    if temperature is not None:
        # Some Ollama models accept temperature via a simple "temperature=" prefix
        prompt = f"temperature={temperature}\n{prompt}"

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"❌ Ollama error:\nSTDERR:\n{result.stderr}\n"
        )

    return result.stdout.strip()