"""
analyzer.py — Gemini-powered viral moment identification for GTA6 AI Content Factory
Sends transcript + heatmap + chat signals to Gemini and returns structured clip candidates.
All timestamps returned as float seconds (start_sec / end_sec).
"""
import os
import time
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


class ViralClip(BaseModel):
    title: str = Field(
        description="Catchy, high-retention YouTube Shorts title — under 60 characters. Specific to GTA VI."
    )
    hook_overlay: str = Field(
        description=(
            "Bold 3-7 word ALL-CAPS on-screen text hook for the first 3 seconds. "
            "Make it shocking, funny, or curiosity-driven."
        )
    )
    alternative_hooks: List[str] = Field(
        description=(
            "Three alternative hook variants using different styles: "
            "1) Curiosity/question, 2) Humour/reaction, 3) Shock/suspense. "
            "Each 3-7 words max."
        ),
        min_length=3,
        max_length=3,
    )
    start_sec: float = Field(
        description="Exact start timestamp in seconds (float). Start 2-3s before the setup begins."
    )
    end_sec: float = Field(
        description=(
            "Exact end timestamp in seconds (float). "
            "CRITICAL: end_sec - start_sec MUST be between 25 and 58 seconds."
        )
    )
    virality_score: int = Field(
        description=(
            "Editorial score 1-100 based on: "
            "hook strength (25%), entertainment value (25%), story completeness (20%), "
            "visual interest (15%), replay potential (15%)."
        )
    )
    confidence_score: int = Field(
        description=(
            "Confidence 1-100 of how certain you are this timestamp range is correct "
            "and self-contained based on the transcript evidence."
        )
    )
    category: str = Field(
        description=(
            "Content category — MUST be one of: "
            "comedy, action, cinematic, discovery, story, fail, unexpected"
        )
    )
    reason: str = Field(
        description="1-2 sentences explaining why viewers will watch to the end and share this clip."
    )
    description: str = Field(
        description=(
            "YouTube Shorts description ~100 characters. "
            "Mention GTA VI and include a call-to-action."
        )
    )
    hashtags: List[str] = Field(
        description=(
            "5-8 relevant hashtags. MUST include #GTA6 and #Shorts. "
            "Example: ['#GTA6', '#Shorts', '#GTA6Funny', '#GTA6Gameplay']"
        ),
        min_length=5,
        max_length=8,
    )


class ClipPlan(BaseModel):
    clips: List[ViralClip]


def format_timestamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _validate_clips(
    clips: List[ViralClip],
    duration_sec: float,
    min_dur: float = 25.0,
    max_dur: float = 58.0,
) -> List[ViralClip]:
    valid: List[ViralClip] = []
    for c in clips:
        dur = c.end_sec - c.start_sec
        if c.start_sec < 0:
            print(f"[Analyzer] Clip '{c.title}' rejected: negative start_sec ({c.start_sec})")
            continue
        if c.end_sec <= c.start_sec:
            print(f"[Analyzer] Clip '{c.title}' rejected: end_sec <= start_sec")
            continue
        if dur < min_dur:
            print(f"[Analyzer] Clip '{c.title}' rejected: too short ({dur:.1f}s < {min_dur}s)")
            continue
        if dur > max_dur:
            trimmed_end = round(c.start_sec + max_dur, 2)
            if duration_sec > 0 and trimmed_end > duration_sec:
                print(f"[Analyzer] Clip '{c.title}' rejected: exceeds source and too long")
                continue
            print(f"[Analyzer] Clip '{c.title}' trimmed to {max_dur}s (was {dur:.1f}s)")
            c = c.model_copy(update={"end_sec": trimmed_end})
        if duration_sec > 0 and c.end_sec > duration_sec + 5:
            print(f"[Analyzer] Clip '{c.title}' rejected: timestamps beyond source duration")
            continue
        valid.append(c)

    valid.sort(key=lambda x: x.virality_score, reverse=True)
    non_overlapping: List[ViralClip] = []
    for c in valid:
        overlap = False
        for existing in non_overlapping:
            if c.start_sec < existing.end_sec and c.end_sec > existing.start_sec:
                overlap = True
                print(
                    f"[Analyzer] Clip '{c.title}' overlaps with '{existing.title}' "
                    f"(score {c.virality_score} vs {existing.virality_score}) — skipped"
                )
                break
        if not overlap:
            non_overlapping.append(c)

    return non_overlapping


def _get_thinking_config(model_name: str, types_mod):
    """Selects the lowest valid thinking setting per Gemini model family."""
    m = model_name.lower()
    if "2.5" in m:
        return types_mod.ThinkingConfig(thinking_budget=0)
    if any(tag in m for tag in ("3.8", "3.7", "gemini-4")):
        return types_mod.ThinkingConfig(thinking_level="low")
    return types_mod.ThinkingConfig(thinking_level="minimal")


def find_viral_clips(
    signals: Dict,
    max_clips: int = 5,
    force_retry: bool = False,
) -> List[Dict]:
    from config import (
        GEMINI_API_KEY, GEMINI_MODELS,
        GEMINI_MAX_RETRIES, GEMINI_BACKOFF_SEC, GEMINI_TEMPERATURE,
        MIN_CLIP_DURATION, MAX_CLIP_DURATION,
    )
    from google import genai
    from google.genai import types

    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

    client = genai.Client(api_key=api_key)

    raw_transcript = signals.get("transcript", "")
    if isinstance(raw_transcript, str):
        transcript_block = raw_transcript
        cue_count = len(raw_transcript.splitlines())
    else:
        transcript_lines = [f"[{cue['start']:.1f}s] {cue['text']}" for cue in raw_transcript]
        transcript_block = "\n".join(transcript_lines)
        cue_count = len(transcript_lines)

    heatmap_items = signals.get("heatmap_peaks", [])
    if heatmap_items and isinstance(heatmap_items[0], dict):
        heatmap_str = ", ".join(
            f"{h['start_sec']:.1f}s (intensity {h['intensity']:.2f})"
            for h in heatmap_items[:20]
        ) or "None available"
    else:
        heatmap_str = "None available"

    chat_items = signals.get("chat_spikes", [])
    if chat_items and isinstance(chat_items[0], dict):
        chat_str = ", ".join(
            f"{c['start_sec']:.1f}s ({c['msgs_in_10s']} msgs/10s, {c['hype_msgs']} hype)"
            for c in chat_items[:20]
        ) or "None available"
    else:
        chat_str = "None available"

    duration_sec = float(signals.get("duration_sec", 0.0))
    video_title = signals.get("title", "GTA VI Gaming VOD")
    duration_note = f" The source video is {duration_sec:.0f} seconds long." if duration_sec > 0 else ""

    schema_hint = json.dumps(ClipPlan.model_json_schema(), indent=2)

    prompt = f"""You are an elite viral short-form gaming content editor specialising in YouTube Shorts, TikTok and Instagram Reels.
Your job is to identify the top {max_clips} most viral, self-contained moments from this gaming stream or VOD.

VIDEO TITLE: {video_title}{duration_note}

YOUTUBE 'MOST REPLAYED' HEATMAP PEAKS (viewers rewound here repeatedly):
{heatmap_str}

LIVE CHAT HYPE SPIKES (moments with most chat activity):
{chat_str}

TIMESTAMPED TRANSCRIPT:
{transcript_block if transcript_block else "[No transcript available — use heatmap and chat signals to estimate viral moments]"}

CONTENT TO PRIORITISE (in order):
1. Funny NPC interactions or unexpected AI behaviour
2. Shocking/unexpected story moments
3. High-energy reactions and fails
4. Hidden discoveries or Easter eggs
5. Intense police chases or action sequences
6. Cinematic or beautiful environment shots
7. Interesting mechanics comparisons
8. Gameplay moments with strong storytelling

HARD RULES (DO NOT VIOLATE):
- Each clip MUST be between {MIN_CLIP_DURATION:.0f} and {MAX_CLIP_DURATION:.0f} seconds (end_sec - start_sec).
- Start 2-3 seconds BEFORE the main action so viewers have context.
- Cut immediately after the punchline, peak reaction, or resolution.
- DO NOT overlap clips — each must cover a unique time range.
- DO NOT fabricate gameplay details not evident from the transcript.
- Include exactly 3 alternative hooks with different emotional tones.
- Hashtags MUST include #GTA6 and #Shorts when relevant.
- Category MUST be one of: comedy, action, cinematic, discovery, story, fail, unexpected

Return ONLY valid JSON matching this JSON Schema:
{schema_hint}
"""

    last_err: Optional[Exception] = None
    all_valid: List[ViralClip] = []

    for model_name in GEMINI_MODELS:
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                print(f"[Analyzer] -> {model_name} (attempt {attempt + 1}) sending {cue_count} cues...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=GEMINI_TEMPERATURE,
                        thinking_config=_get_thinking_config(model_name, types),
                        http_options=types.HttpOptions(timeout=45_000),
                    ),
                )
                plan = ClipPlan.model_validate_json(response.text)
                validated = _validate_clips(
                    plan.clips, duration_sec, MIN_CLIP_DURATION, MAX_CLIP_DURATION
                )

                if validated:
                    all_valid = validated
                    print(
                        f"[Analyzer] OK {model_name} returned {len(plan.clips)} clips, "
                        f"{len(validated)} passed validation."
                    )
                    break
                else:
                    print(
                        f"[Analyzer] {model_name} returned {len(plan.clips)} clips but NONE "
                        f"passed validation. Retrying..."
                    )
                    if attempt == 0:
                        continue
            except (ValidationError, json.JSONDecodeError) as je:
                last_err = je
                print(f"[Analyzer] {model_name}: JSON/Validation error — {je}")
                break
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if any(x in err_str for x in ("quota", "rate", "429", "503", "timeout")):
                    wait = GEMINI_BACKOFF_SEC * (2 ** attempt)
                    print(f"[Analyzer] {model_name}: transient error ({e}). Waiting {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[Analyzer] {model_name}: permanent error ({type(e).__name__}: {e})")
                    break

        if all_valid:
            break

    if not all_valid:
        raise RuntimeError(
            f"All Gemini models failed or returned no valid clips. Last error: {last_err}"
        )

    all_valid.sort(key=lambda x: x.virality_score, reverse=True)

    results = []
    for c in all_valid[:max_clips]:
        d = c.model_dump()
        d["start_hms"] = format_timestamp(c.start_sec)
        d["end_hms"] = format_timestamp(c.end_sec)
        d["duration"] = round(c.end_sec - c.start_sec, 1)
        results.append(d)

    return results


if __name__ == "__main__":
    from extractor import fetch_video_signals, parse_vtt_dedup, WORK_DIR

    test_url = input(
        "Paste YouTube URL to analyze (or press Enter to reuse cached workspace): "
    ).strip()

    if test_url:
        sigs = fetch_video_signals(test_url)
    else:
        info_files = list(WORK_DIR.glob("target*.info.json"))
        if not info_files:
            print("No cached workspace found. Please provide a URL.")
            exit(1)
        info = json.loads(info_files[0].read_text(encoding="utf-8"))
        vtt_list = list(WORK_DIR.glob("target*.vtt"))
        cues = parse_vtt_dedup(vtt_list[0]) if vtt_list else []
        sigs = {
            "title": info.get("title", "Gaming VOD"),
            "duration_sec": float(info.get("duration") or 0.0),
            "heatmap_peaks": [],
            "chat_spikes": [],
            "cues": cues,
            "transcript": "\n".join(f"[{c['start']:.1f}s] {c['text']}" for c in cues),
        }

    clips = find_viral_clips(sigs, max_clips=5)
    print("\n=== TOP VIRAL CLIPS IDENTIFIED ===")
    for i, clip in enumerate(clips, 1):
        print(f"\n#{i} [{clip['start_hms']} -> {clip['end_hms']} | {clip['duration']}s | "
              f"Score: {clip['virality_score']}/100 | Confidence: {clip['confidence_score']}/100]")
        print(f"   Category:  {clip['category'].upper()}")
        print(f"   Title:     {clip['title']}")
        print(f"   Hook:      {clip['hook_overlay']}")
        print(f"   Alt Hooks: {clip['alternative_hooks']}")
        print(f"   Hashtags:  {' '.join(clip['hashtags'])}")
        print(f"   Why:       {clip['reason']}")