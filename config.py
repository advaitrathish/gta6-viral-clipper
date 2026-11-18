"""
config.py — Central configuration for GTA6 AI Content Factory
All secrets come from .env — never hardcoded here.
"""
import os
import platform
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
WORK_DIR   = BASE_DIR / "workspace"
CACHE_DIR  = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output_clips"
LOGS_DIR   = BASE_DIR / "logs"

for _d in (WORK_DIR, CACHE_DIR, OUTPUT_DIR, LOGS_DIR):
    _d.mkdir(exist_ok=True)

COOKIES_FILE = BASE_DIR / "cookies.txt"

# ---------------------------------------------------------------------------
# Gemini Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

_env_models = os.getenv("GEMINI_MODELS", "").strip()
GEMINI_MODELS = (
    [m.strip() for m in _env_models.split(",") if m.strip()]
    if _env_models
    else [
        "gemini-4-hydrogen",       # Replace with exact hydrogen ID from list above
        "gemini-3.8-flash",        # Workhorse fallback 1
        "gemini-4-carbon",         # Replace with exact carbon ID from list above
        "gemini-3.6-flash",        # Fast fallback 3
        "gemini-3.5-flash-lite",   # Low-latency fallback 4
    ]
)
GEMINI_MAX_RETRIES  = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_BACKOFF_SEC  = float(os.getenv("GEMINI_BACKOFF_SEC", "2.0"))
GEMINI_TEMPERATURE  = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))

# ---------------------------------------------------------------------------
# Clip Validation Constraints
# ---------------------------------------------------------------------------
MIN_CLIP_DURATION = float(os.getenv("MIN_CLIP_DURATION", "25.0"))
MAX_CLIP_DURATION = float(os.getenv("MAX_CLIP_DURATION", "58.0"))

# ---------------------------------------------------------------------------
# Telegram Bot (optional)
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USER = os.getenv("TELEGRAM_ALLOWED_USER", "")

# ---------------------------------------------------------------------------
# GitHub Actions (optional)
# ---------------------------------------------------------------------------
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO_OWNER = os.getenv("GITHUB_REPO_OWNER", "")
GITHUB_REPO_NAME  = os.getenv("GITHUB_REPO_NAME", "gta6-clipper")

# ---------------------------------------------------------------------------
# YouTube Publishing (optional)
# ---------------------------------------------------------------------------
YT_CLIENT_SECRETS = BASE_DIR / "credentials.json"
YT_TOKEN_FILE     = BASE_DIR / "token.json"

# ---------------------------------------------------------------------------
# Font Discovery (Windows + Linux compatible)
# ---------------------------------------------------------------------------
def find_font() -> str:
    """Auto-detect a bold font path suitable for FFmpeg drawtext / ASS."""
    system = platform.system()
    candidates = []
    if system == "Windows":
        win_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates = [
            win_fonts / "arialbd.ttf",   # Arial Bold
            win_fonts / "arial.ttf",
            win_fonts / "impact.ttf",
            win_fonts / "verdanab.ttf",
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf"),
            Path("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
        ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Last resort: ask fontconfig
    fc = shutil.which("fc-match")
    if fc:
        import subprocess
        try:
            result = subprocess.run(
                [fc, "--format=%{file}", ":weight=bold"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    return "Arial"   # FFmpeg built-in fallback

FONT_PATH = find_font()

# ---------------------------------------------------------------------------
# Audio normalization targets
# ---------------------------------------------------------------------------
AUDIO_LOUDNESS_TARGET  = float(os.getenv("AUDIO_LOUDNESS_TARGET",  "-14.0"))   # LUFS
AUDIO_TRUE_PEAK_TARGET = float(os.getenv("AUDIO_TRUE_PEAK_TARGET", "-1.0"))    # dBTP

# ---------------------------------------------------------------------------
# Debug / logging
# ---------------------------------------------------------------------------
DEBUG = os.getenv("DEBUG", "0").strip().lower() in ("1", "true", "yes")