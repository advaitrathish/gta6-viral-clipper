"""
extractor.py — Signal extraction for GTA6 AI Content Factory
Fetches: metadata, YouTube Most Replayed heatmap, English subtitles/auto-captions,
         live chat replay, hype keyword analysis.
All timestamps are standardized to floating-point seconds throughout.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from config import COOKIES_FILE, WORK_DIR

HYPE_PATTERN = re.compile(
    r"\b(w+|l+|lol+|lmao+|lmfao+|wtf+|omg+|no way|clip|insane|crazy|goat|sheesh|cooked|wild|rip|gg+)\b|[🔥💀😂😭🤯😱🚀]",
    re.IGNORECASE,
)


def hms_to_seconds(hms: str) -> float:
    hms = hms.strip().replace(",", ".")
    parts = [float(p) for p in hms.split(":")]
    if len(parts) == 3:
        return round(parts[0] * 3600 + parts[1] * 60 + parts[2], 3)
    if len(parts) == 2:
        return round(parts[0] * 60 + parts[1], 3)
    return round(parts[0], 3)


def seconds_to_hms(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _find_ytdlp() -> List[str]:
    scripts_dir = Path(sys.executable).parent
    for name in ("yt-dlp.exe", "yt-dlp"):
        candidate = scripts_dir / name
        if candidate.exists():
            return [str(candidate)]
    on_path = shutil.which("yt-dlp")
    if on_path:
        return [on_path]
    return [sys.executable, "-m", "yt_dlp"]


def get_yt_dlp_base() -> List[str]:
    cmd = _find_ytdlp() + ["--no-warnings", "--no-playlist", "--remote-components", "ejs:github"]
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 10:
        cmd.extend(["--cookies", str(COOKIES_FILE)])
    return cmd


def _strip_rolling_overlap(prev_text: str, curr_text: str) -> str:
    prev_words = prev_text.split()
    curr_words = curr_text.split()
    max_overlap = min(len(prev_words), len(curr_words))
    for k in range(max_overlap, 0, -1):
        if [w.lower() for w in prev_words[-k:]] == [w.lower() for w in curr_words[:k]]:
            remainder = " ".join(curr_words[k:]).strip()
            return remainder
    return curr_text


def parse_vtt_dedup(vtt_path) -> List[Dict]:
    if vtt_path is None:
        return []
    vtt_path = Path(vtt_path)
    if not vtt_path.exists():
        return []

    raw_lines = vtt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    time_re = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s-->\s(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    )

    raw_cues: List[Dict] = []
    curr_s: Optional[float] = None
    curr_e: Optional[float] = None
    curr_txt: List[str] = []

    def _flush():
        nonlocal curr_s, curr_e, curr_txt
        if curr_s is not None and curr_e is not None and curr_txt:
            joined = " ".join(curr_txt)
            cleaned = re.sub(r"<[^>]+>", "", joined)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                raw_cues.append({"start": curr_s, "end": curr_e, "text": cleaned})

    for line in raw_lines:
        stripped = line.strip()
        m = time_re.search(stripped)
        if m:
            _flush()
            curr_s = hms_to_seconds(m.group(1))
            curr_e = hms_to_seconds(m.group(2))
            curr_txt = []
        elif (
            stripped
            and not stripped.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE", "REGION"))
            and not stripped.isdigit()
        ):
            curr_txt.append(stripped)

    _flush()
    raw_cues.sort(key=lambda c: c["start"])

    deduped: List[Dict] = []
    last_full_text = ""
    for cue in raw_cues:
        text = cue["text"]
        if text.lower() == last_full_text.lower():
            continue
        trimmed = _strip_rolling_overlap(last_full_text, text)
        last_full_text = text
        if trimmed:
            deduped.append({"start": cue["start"], "end": cue["end"], "text": trimmed})

    return deduped


def _pick_best_vtt(vtt_files: List[Path]) -> Optional[Path]:
    if not vtt_files:
        return None
    for f in vtt_files:
        if ".en-orig." in f.name:
            return f
    for f in vtt_files:
        if re.search(r"\.en[.-]", f.name):
            return f
    return vtt_files[0]


def fetch_video_signals(url: str) -> Dict:
    for old in WORK_DIR.glob("target*"):
        old.unlink(missing_ok=True)

    out_template = str(WORK_DIR / "target")
    cmd = get_yt_dlp_base() + [
        "--skip-download",
        "--ignore-no-formats-error",
        "--write-info-json",
        "--write-sub",
        "--write-auto-sub",
        "--sub-langs", "en-orig,en.*,en,live_chat",
        "--sub-format", "vtt/json",
        "-o", out_template,
        url,
    ]
    print("[Extractor] Fetching metadata, transcript, and live chat replay...")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(cmd, check=False, capture_output=False, env=env)

    info_files = list(WORK_DIR.glob("target*.info.json"))
    title = "Gaming VOD"
    video_id = ""
    duration_sec = 0.0
    heatmap_peaks: List[Dict] = []

    if info_files:
        try:
            info = json.loads(info_files[0].read_text(encoding="utf-8", errors="ignore"))
            title = info.get("title", title)
            video_id = info.get("id", "")
            duration_sec = float(info.get("duration") or 0.0)
            for entry in info.get("heatmap") or []:
                val = float(entry.get("value", 0.0))
                if val >= 0.60:
                    heatmap_peaks.append({
                        "start_sec": round(float(entry.get("start_time", 0.0)), 2),
                        "end_sec": round(float(entry.get("end_time", 0.0)), 2),
                        "intensity": round(val, 3),
                    })
            heatmap_peaks.sort(key=lambda x: x["intensity"], reverse=True)
        except Exception as e:
            print(f"[Extractor] Warning: could not parse info.json — {e}")

    chat_files = list(WORK_DIR.glob("target*.live_chat.json"))
    chat_spikes: List[Dict] = []

    if chat_files:
        print("[Extractor] Analysing live chat replay velocity...")
        buckets: Dict = defaultdict(lambda: {"count": 0, "hype": 0, "samples": []})
        try:
            with chat_files[0].open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        action = data.get("replayChatItemAction", {})
                        offset_ms = int(action.get("videoOffsetTimeMsec", -1))
                        if offset_ms < 0:
                            continue
                        bucket_sec = float(int((offset_ms / 1000.0) // 10) * 10)
                        for act in action.get("actions", []):
                            item = act.get("addChatItemAction", {}).get("item", {})
                            renderer = item.get("liveChatTextMessageRenderer", {})
                            runs = renderer.get("message", {}).get("runs", [])
                            msg = "".join(r.get("text", "") for r in runs).strip()
                            if not msg:
                                continue
                            buckets[bucket_sec]["count"] += 1
                            if HYPE_PATTERN.search(msg):
                                buckets[bucket_sec]["hype"] += 1
                            if len(buckets[bucket_sec]["samples"]) < 4:
                                buckets[bucket_sec]["samples"].append(msg[:50])
                    except Exception:
                        continue
        except Exception as e:
            print(f"[Extractor] Warning: could not parse live chat — {e}")

        for b_sec, stats in buckets.items():
            chat_spikes.append({
                "start_sec": b_sec,
                "msgs_in_10s": stats["count"],
                "hype_msgs": stats["hype"],
                "score": stats["count"] + (stats["hype"] * 2),
                "sample_chat": " | ".join(stats["samples"]),
            })
        chat_spikes.sort(key=lambda x: x["score"], reverse=True)
        chat_spikes = chat_spikes[:25]

    vtt_files = list(WORK_DIR.glob("target*.vtt"))
    best_vtt = _pick_best_vtt(vtt_files)
    cues = parse_vtt_dedup(best_vtt) if best_vtt else []

    if not cues:
        print("[Extractor] No subtitles found — Gemini will analyze with heatmap/chat signals only.")
    else:
        print(f"[Extractor] Parsed {len(cues)} deduplicated subtitle cues.")

    transcript_text = "\n".join(f"[{c['start']:.1f}s] {c['text']}" for c in cues)

    return {
        "url": url,
        "video_id": video_id,
        "title": title,
        "duration_sec": duration_sec,
        "heatmap_peaks": heatmap_peaks[:25],
        "chat_spikes": chat_spikes,
        "cues": cues,
        "transcript": transcript_text,
    }


if __name__ == "__main__":
    test_url = input("Paste a YouTube gaming video/stream VOD URL to test: ").strip()
    result = fetch_video_signals(test_url)
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Title:          {result['title']} ({result['duration_sec']}s)")
    print(f"Video ID:       {result['video_id']}")
    print(f"Heatmap Peaks:  {len(result['heatmap_peaks'])} -> Top 3: {result['heatmap_peaks'][:3]}")
    print(f"Chat Spikes:    {len(result['chat_spikes'])} -> Top 3: {result['chat_spikes'][:3]}")
    print(f"Subtitle Cues:  {len(result['cues'])}")
    print("Sample Transcript:\n" + "\n".join(result["transcript"].splitlines()[:8]))