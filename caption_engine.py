"""
caption_engine.py — ASS subtitle generation for GTA6 AI Content Factory
Converts cue dicts into Advanced SubStation Alpha (.ass) subtitle files
suitable for FFmpeg burning via -vf ass=file.ass.
"""
import textwrap
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# ASS colour constants (BGR hex, alpha = 00 = fully opaque)
# ---------------------------------------------------------------------------
WHITE   = "&H00FFFFFF"
BLACK   = "&H00000000"
YELLOW  = "&H0000FFFF"   # High-contrast yellow for action/highlights
SHADOW  = "&H80000000"   # Semi-transparent black shadow
OUTLINE = "&H00000000"

# Liberation Sans is metric-identical to Arial and pre-installed on Linux runners
DEFAULT_FONT = "Liberation Sans"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seconds_to_ass(sec: float) -> str:
    """Convert float seconds to ASS timestamp h:mm:ss.cc"""
    sec = max(0.0, sec)
    h   = int(sec // 3600)
    m   = int((sec % 3600) // 60)
    s   = int(sec % 60)
    cs  = int(round((sec - int(sec)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _wrap_text(text: str, max_chars: int = 26) -> str:
    """Word-wrap long lines into punchy 1-2 line chunks inside the Shorts safe zone."""
    lines = textwrap.wrap(text.strip(), width=max_chars)
    return r"\N".join(lines[:3])   # Cap at 3 lines max so captions never cover gameplay


def _escape_ass(text: str) -> str:
    """Escape curly braces (ASS tag delimiters) in plain text."""
    return text.replace("{", "\\{").replace("}", "\\}")


# ---------------------------------------------------------------------------
# ASS header factory
# ---------------------------------------------------------------------------

def _build_ass_header(
    resolution_x: int = 1080,
    resolution_y: int = 1920,
    font_name: str = DEFAULT_FONT,
    font_size: int = 72,
    preset: str = "default",
) -> str:
    """Generate ASS [Script Info] + [V4+ Styles] header."""

    if preset in ("comedy", "fail", "unexpected"):
        bold        = -1
        outline_px  = 5
        shadow_px   = 3
        primary_col = WHITE
        outline_col = BLACK
    elif preset == "action":
        bold        = -1
        outline_px  = 5
        shadow_px   = 4
        primary_col = YELLOW
        outline_col = BLACK
    elif preset == "cinematic":
        bold        = 0
        outline_px  = 3
        shadow_px   = 2
        primary_col = WHITE
        outline_col = "&H80000000"
    else:  # discovery / story / default
        bold        = -1
        outline_px  = 5
        shadow_px   = 3
        primary_col = WHITE
        outline_col = BLACK

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {resolution_x}
PlayResY: {resolution_y}
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_col},{WHITE},{outline_col},{SHADOW},{bold},0,0,0,100,100,0,0,1,{outline_px},{shadow_px},2,70,70,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ass_captions(
    cues: List[Dict],
    output_path: Path,
    clip_start_sec: float,
    clip_end_sec: float,
    font_name: str = DEFAULT_FONT,
    font_size: int = 72,
    preset: str = "default",
) -> bool:
    """
    Write an ASS subtitle file for a clip time range.
    """
    if not cues:
        return False

    clip_cues = [
        c for c in cues
        if c["end"] > clip_start_sec and c["start"] < clip_end_sec
    ]

    if not clip_cues:
        return False

    header   = _build_ass_header(1080, 1920, font_name, font_size, preset)
    dialogue = []

    for cue in clip_cues:
        rel_start = max(0.0, cue["start"] - clip_start_sec)
        rel_end   = min(clip_end_sec - clip_start_sec, cue["end"] - clip_start_sec)
        if rel_end <= rel_start:
            continue

        ass_start = _seconds_to_ass(rel_start)
        ass_end   = _seconds_to_ass(rel_end)
        text      = _wrap_text(_escape_ass(cue["text"]), max_chars=26)

        dialogue.append(f"Dialogue: 0,{ass_start},{ass_end},Default,,0,0,0,,{text}")

    if not dialogue:
        return False

    output_path.write_text(header + "\n".join(dialogue) + "\n", encoding="utf-8")
    return True


def generate_hook_ass(
    hook_text: str,
    output_path: Path,
    display_duration: float = 3.5,
    font_name: str = DEFAULT_FONT,
    preset: str = "default",
) -> bool:
    """
    Write an ASS file containing a single large hook overlay at the top of the clip.
    Displayed for the first display_duration seconds.
    """
    wrapped  = _wrap_text(hook_text.upper(), max_chars=20)
    ass_hook = _escape_ass(wrapped)
    content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,{font_name},84,{YELLOW},{WHITE},{BLACK},{SHADOW},-1,0,0,0,100,100,2,0,1,6,4,8,70,70,190,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,{_seconds_to_ass(0.0)},{_seconds_to_ass(display_duration)},Hook,,0,0,0,,{ass_hook}
"""
    output_path.write_text(content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cues = [
        {"start": 10.0, "end": 13.5, "text": "Oh no, the police are coming!"},
        {"start": 13.5, "end": 17.0, "text": "And he just runs straight into them!"},
        {"start": 17.0, "end": 21.0, "text": "Five star wanted level already!"},
    ]
    out = Path("test_captions.ass")
    ok  = generate_ass_captions(test_cues, out, 10.0, 25.0, preset="comedy")
    print(f"Caption file written: {ok} → {out}")
    print(out.read_text())