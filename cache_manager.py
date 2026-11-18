"""
cache_manager.py — URL-based caching for GTA6 AI Content Factory
Avoids re-extracting transcripts and re-running Gemini analysis.
Simple JSON file cache keyed by URL hash.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from config import CACHE_DIR


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _url_key(url: str) -> str:
    """SHA256 hash of the URL as the cache key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _cache_path(url: str, suffix: str) -> Path:
    return CACHE_DIR / f"{_url_key(url)}_{suffix}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_signals(url: str) -> Optional[dict]:
    """Load cached extraction signals for a URL. Returns None if not found."""
    p = _cache_path(url, "signals")
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            print(f"[Cache] ✓ Loaded signals from cache (key: {_url_key(url)})")
            return data
        except Exception as e:
            print(f"[Cache] ⚠ Could not read signals cache: {e}")
    return None


def save_signals(url: str, signals: dict) -> None:
    """Persist extraction signals for a URL."""
    p = _cache_path(url, "signals")
    try:
        p.write_text(json.dumps(signals, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"[Cache] Saved signals → {p.name}")
    except Exception as e:
        print(f"[Cache] ⚠ Could not save signals: {e}")


def load_clips(url: str) -> Optional[list]:
    """Load cached Gemini clip analysis for a URL. Returns None if not found."""
    p = _cache_path(url, "clips")
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            print(f"[Cache] ✓ Loaded clips from cache (key: {_url_key(url)})")
            return data
        except Exception as e:
            print(f"[Cache] ⚠ Could not read clips cache: {e}")
    return None


def save_clips(url: str, clips: list) -> None:
    """Persist Gemini clip analysis for a URL."""
    p = _cache_path(url, "clips")
    try:
        p.write_text(json.dumps(clips, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"[Cache] Saved clips → {p.name}")
    except Exception as e:
        print(f"[Cache] ⚠ Could not save clips: {e}")


def clear_cache(url: str) -> None:
    """Remove all cached data for a URL."""
    for suffix in ("signals", "clips"):
        p = _cache_path(url, suffix)
        p.unlink(missing_ok=True)
    print(f"[Cache] Cleared cache for key {_url_key(url)}")


def cache_exists(url: str) -> bool:
    """Return True if any cached data exists for the URL."""
    return (
        _cache_path(url, "signals").exists()
        or _cache_path(url, "clips").exists()
    )


# ---------------------------------------------------------------------------
# CLI utility
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        clear_cache(sys.argv[1])
        print("Cache cleared.")
    else:
        files = list(CACHE_DIR.glob("*.json"))
        print(f"Cache directory: {CACHE_DIR}")
        print(f"Cached entries:  {len(files)}")
        for f in files:
            size = f.stat().st_size
            mtime = time.ctime(f.stat().st_mtime)
            print(f"  {f.name}  ({size} bytes, {mtime})")
