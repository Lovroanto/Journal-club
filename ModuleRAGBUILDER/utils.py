# ModuleRAGBUILDER/utils.py
"""
Small utility functions used across the ModuleRAGBUILDER package.
Keep functions small and dependency-free.
"""

import json
import re
import time
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional, Any, Dict

# Basic configuration
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
REQUESTS_TIMEOUT = 20  # seconds


def clean_filename(s: str, max_len: int = 120) -> str:
    """Make a filesystem-safe short filename."""
    if s is None:
        s = ""
    # remove weird characters, keep basic punctuation
    s = re.sub(r"[^\w\s\-\(\)\.\[\]]+", "", str(s))
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len]


def stable_similarity(a: str, b: str) -> float:
    """Simple 0..1 similarity using difflib.SequenceMatcher."""
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def ensure_str(x: Optional[Any]) -> Optional[str]:
    """If x is a list, return first element; else coerce to str if possible."""
    if x is None:
        return None
    if isinstance(x, list) and x:
        return x[0]
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return None


def makedirs(p: Path) -> None:
    """Create directories if missing."""
    p.mkdir(parents=True, exist_ok=True)


# -----------------------
# Simple JSON disk cache
# -----------------------
class JsonCache:
    """
    Very small JSON-on-disk cache.
    Keys are turned into filenames via clean_filename.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        makedirs(cache_dir)

    def _path(self, key: str) -> Path:
        name = clean_filename(key)
        return self.cache_dir / (name + ".json")

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, key: str, value) -> None:
        p = self._path(key)
        try:
            p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # swallow write errors (disk full / permissions)
            pass


# -----------------------
# Small logging helpers
# -----------------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def info(msg: str) -> None:
    print(f"[INFO {now_ts()}] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN {now_ts()}] {msg}")


def err(msg: str) -> None:
    print(f"[ERROR {now_ts()}] {msg}")
