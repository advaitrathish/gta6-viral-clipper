"""
live_chat_watcher.py — Real-time YouTube Live Chat Velocity & Hype-Burst Clipper
Supports English + Hindi/Hinglish streams (IShowSpeed, Techno Gamerz, GamerFleet, Triggered Insaan, etc.)
Includes audio peak reaction alignment, faster-whisper live captions, and duplicate clip protection.
"""
import argparse
import json
import logging
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from config import BASE_DIR, OUTPUT_DIR
from analyzer import find_viral_clips
from clipper import render_vertical_short, get_ffmpeg_exe, CATEGORY_TO_PRESET
from quality_checker import check_video

WORKSPACE_DIR = BASE_DIR / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
COOKIES_FILE = BASE_DIR / "cookies.txt"
HISTORY_FILE = WORKSPACE_DIR / "clip_history.json"

logger = logging.getLogger("live_chat_watcher")

# English + Hindi/Hinglish viral reaction tokens (case-insensitive)
HYPE_PATTERN = re.compile(
    r"\b(w+|l+|lmao+|lmfao+|lol+|wtf+|omg+|no\s*way|what+|bro+|clip(\s*it)?|gg+|ez+|"
    r"op+|bhai+|arey+|abe+|kya+|pagal|gajab|mast|khatarnak|haha+|xd+|💀|😭|🔥|😱|😂)\b",
    re.IGNORECASE,
)


def _format_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _is_already_clipped(video_id: str, bucket_sec: float) -> bool:
    """Prevent duplicate clips of the same livestream moment across repeated runs."""
    bucket_key = f"{video_id}:{int(bucket_sec // 60)}"
    if not HISTORY_FILE.exists():
        return False
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return bucket_key in data.get("clipped", [])
    except Exception:
        return False


def _mark_clipped(video_id: str, bucket_sec: float) -> None:
    bucket_key = f"{video_id}:{int(bucket_sec // 60)}"
    data = {"clipped": []}
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    clipped = data.get("clipped", [])
    if bucket_key not in clipped:
        clipped.append(bucket_key)
    data["clipped"] = clipped[-500:]
    HISTORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_live_metadata(url: str) -> dict:
    """Fetch video_id, title, is_live status, and release_timestamp from YouTube."""
    cmd = ["yt-dlp", "--dump-json", "--no-playlist", "--no-warnings", "--js-runtimes", "deno", "--remote-components", "ejs:github"]
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 0:
        cmd += ["--cookies", str(COOKIES_FILE)]
    cmd.append(url)

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"yt-dlp metadata failed: {res.stderr.strip()[-600:]}")
    data = json.loads(res.stdout.splitlines()[0])
    return {
        "video_id": data.get("id"),
        "title": data.get("title", "Live Stream"),
        "channel": data.get("uploader", ""),
        "is_live": bool(data.get("is_live")),
        "release_timestamp": data.get("release_timestamp"),
        "webpage_url": data.get("webpage_url", url),
    }


def collect_live_chat(url: str, duration_sec: int = 480) -> Path:
    """Record live chat JSONL via yt-dlp for `duration_sec` seconds."""
    chat_prefix = WORKSPACE_DIR / "live_chat_capture"
    for old in WORKSPACE_DIR.glob("live_chat_capture*"):
        old.unlink(missing_ok=True)

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--sub-langs", "live_chat",
        "--remote-components", "ejs:github",
        "-o", str(chat_prefix),
    ]
    if COOKIES_FILE.exists():
        cmd += ["--cookies", str(COOKIES_FILE)]
    cmd.append(url)

    logger.info(f"Listening to live chat for {duration_sec}s...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        proc.wait(timeout=duration_sec)
    except subprocess.TimeoutExpired:
        logger.info("Capture window reached — stopping live chat listener cleanly.")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    files = list(WORKSPACE_DIR.glob("live_chat_capture*.live_chat.json*"))
    if not files:
        raise RuntimeError("No live_chat.json captured (stream may have chat disabled or ended).")
    return files[0]


def detect_chat_spikes(chat_file: Path, stream_start_ts: float | None, top_k: int = 2) -> list[dict]:
    """Parse YouTube live_chat JSONL and score 10-second buckets by UNIQUE hype authors."""
    buckets = defaultdict(lambda: {"authors": set(), "hype_authors": set(), "messages": []})
    first_msg_sec = None

    with chat_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            actions = (
                obj.get("replayChatItemAction", {}).get("actions")
                or obj.get("addChatItemAction")
                or []
            )
            if isinstance(actions, dict):
                actions = [actions]

            for act in actions:
                item = act.get("addChatItemAction", {}).get("item", {})
                renderer = (
                    item.get("liveChatTextMessageRenderer")
                    or item.get("liveChatPaidMessageRenderer")
                )
                if not renderer:
                    continue

                usec = renderer.get("timestampUsec")
                if not usec:
                    continue
                msg_sec = int(usec) / 1_000_000.0
                if first_msg_sec is None:
                    first_msg_sec = msg_sec

                author = renderer.get("authorExternalChannelId") or renderer.get("authorName", {}).get("simpleText", "anon")
                runs = renderer.get("message", {}).get("runs", [])
                text = "".join(r.get("text", "") for r in runs).strip()
                if not text:
                    continue

                bucket_idx = int(msg_sec // 10) * 10
                b = buckets[bucket_idx]
                b["authors"].add(author)
                if HYPE_PATTERN.search(text):
                    b["hype_authors"].add(author)
                if len(b["messages"]) < 25:
                    b["messages"].append(text)

    if not buckets:
        return []

    base_ts = float(stream_start_ts) if stream_start_ts else float(first_msg_sec or 0)
    scored = []
    for b_sec, info in sorted(buckets.items()):
        unique_total = len(info["authors"])
        unique_hype = len(info["hype_authors"])
        if unique_hype < 2 and unique_total < 4:
            continue
        score = (unique_hype * 2.5) + unique_total
        offset_sec = max(0.0, b_sec - base_ts)
        scored.append({
            "bucket_sec": b_sec,
            "offset_sec": offset_sec,
            "score": score,
            "unique_total": unique_total,
            "unique_hype": unique_hype,
            "sample_chat": info["messages"],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for cand in scored:
        if any(abs(cand["offset_sec"] - s["offset_sec"]) < 75 for s in selected):
            continue
        start_s = max(0.0, cand["offset_sec"] - 35.0)
        end_s = start_s + 50.0
        cand["start_ts"] = _format_ts(start_s)
        cand["end_ts"] = _format_ts(end_s)
        selected.append(cand)
        if len(selected) >= top_k:
            break

    return selected


def slice_live_segment(url: str, start_ts: str, end_ts: str, out_path: Path, seconds_ago: float = 60.0) -> Path:
    """Resolve HLS m3u8 live URL via yt-dlp -g and slice 50s from the live DVR buffer with FFmpeg."""
    if out_path.exists():
        out_path.unlink()

    url_cmd = [
        "yt-dlp",
        "-g",
        "-f", "best[height<=1080][protocol!*=dash]/best[protocol!*=dash]/best",
        "--remote-components", "ejs:github",
    ]
    if COOKIES_FILE.exists():
        url_cmd += ["--cookies", str(COOKIES_FILE)]
    url_cmd.append(url)

    u_res = subprocess.run(url_cmd, capture_output=True, text=True, timeout=60)
    stream_urls = [line.strip() for line in u_res.stdout.splitlines() if line.strip().startswith("http")]
    if u_res.returncode != 0 or not stream_urls:
        raise RuntimeError(f"Failed to resolve HLS live URL: {u_res.stderr.strip()[:300]}")

    ffmpeg_bin = get_ffmpeg_exe()
    seek_back = f"-{max(55, int(seconds_ago + 35))}"
    ff_cmd = [
        ffmpeg_bin, "-y",
        "-sseof", seek_back,
        "-i", stream_urls[0],
        "-t", "50",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]

    ff_res = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=180)
    if ff_res.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"FFmpeg HLS slice failed: {ff_res.stderr.strip()[-300:]}")
    return out_path


def _align_to_audio_peak(raw_path: Path, total_sec: float = 50.0) -> tuple[float, float]:
    """Find the loudest streamer reaction peak in the slice and frame [peak-5s, peak+35s]."""
    try:
        ff_exe = get_ffmpeg_exe()
        res = subprocess.run(
            [
                ff_exe, "-i", str(raw_path),
                "-af", "asetnsamples=48000,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=25,
        )
        sec_rms: list[tuple[float, float]] = []
        cur_t = 0.0
        for line in res.stderr.splitlines():
            m_t = re.search(r"pts_time:([0-9.]+)", line)
            if m_t:
                cur_t = float(m_t.group(1))
            m_v = re.search(r"RMS_level=(-?[0-9.]+)", line)
            if m_v:
                val = float(m_v.group(1))
                if val > -90.0:
                    sec_rms.append((cur_t, val))

        if sec_rms:
            peak_t, _ = max(sec_rms, key=lambda x: x[1])
            start_sec = max(0.0, round(peak_t - 5.0, 1))
            end_sec = min(total_sec, round(start_sec + 40.0, 1))
            if end_sec - start_sec >= 20.0:
                logger.info(f"Audio peak aligned: {start_sec}s -> {end_sec}s (peak @ {peak_t:.1f}s)")
                return start_sec, end_sec
    except Exception as e:
        logger.warning(f"Audio alignment skipped: {e}")
    return 0.0, total_sec


def _transcribe_live_cues(raw_path: Path) -> list[dict]:
    """Generate timestamped caption cues from live audio using faster-whisper (supports English + Hindi)."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(raw_path), beam_size=1, vad_filter=True)
        cues = []
        for seg in segments:
            txt = seg.text.strip()
            if txt:
                cues.append({
                    "start": round(float(seg.start), 2),
                    "end": round(float(seg.end), 2),
                    "text": txt,
                })
        logger.info(f"Transcribed {len(cues)} live caption segments via faster-whisper.")
        return cues
    except Exception as e:
        logger.warning(f"Live caption transcription skipped: {e}")
        return []


def _generate_live_clip_meta(spike: dict, stream_meta: dict) -> dict:
    """Pass a structured signals dict into analyzer.find_viral_clips for Gemini hook/title generation."""
    synthetic_transcript = (
        f"[00:00:00 -> 00:00:50] LIVE CHAT EXPLOSION "
        f"({spike['unique_hype']} hype viewers / {spike['unique_total']} total active): "
        + " | ".join(spike["sample_chat"][:15])
    )
    fallback = {
        "start_time": "00:00:00",
        "end_time": "00:00:50",
        "duration_sec": 50.0,
        "virality_score": min(99, int(75 + spike["unique_hype"])),
        "confidence": 88,
        "category": "comedy",
        "title": f"{stream_meta['channel']} Craziest Live Moment",
        "hook_overlay": "CHAT WENT CRAZY HERE",
        "hashtags": ["#GTA6", "#Shorts", "#Live", "#Gaming"],
        "reason": f"Live chat spiked with {spike['unique_hype']} unique viewers reacting simultaneously.",
    }
    try:
        signals = {
            "video_id": stream_meta.get("video_id", "live"),
            "title": stream_meta.get("title", "Live Stream"),
            "channel": stream_meta.get("channel", ""),
            "duration": 50,
            "transcript": synthetic_transcript,
            "heatmap": [],
            "comments": spike["sample_chat"][:15],
        }
        clips = find_viral_clips(signals=signals, max_clips=1)
        if clips:
            c = dict(clips[0])
            c["start_time"] = "00:00:00"
            c["end_time"] = "00:00:50"
            c["duration_sec"] = 50.0
            return c
    except Exception as e:
        logger.warning(f"Gemini metadata fallback triggered: {e}")
    return fallback


def run_live_monitor(url: str, listen_sec: int = 480, top_n: int = 2) -> dict:
    """Full live chat burst pipeline -> returns same result format as pipeline.run_pipeline()."""
    t0 = time.time()
    meta = get_live_metadata(url)
    if not meta["is_live"]:
        logger.warning("URL is not currently live — falling back to standard VOD pipeline.")
        from pipeline import run_pipeline
        from drive_uploader import upload_clip_to_drive
        vod_res = run_pipeline(url=url, max_clips=5, top_n=top_n)
        for r in vod_res.get("results", []):
            if r.get("success") and r.get("path"):
                r["drive"] = upload_clip_to_drive(Path(r["path"]))
        return vod_res

    chat_file = collect_live_chat(meta["webpage_url"], duration_sec=listen_sec)
    spikes = detect_chat_spikes(chat_file, meta.get("release_timestamp"), top_k=top_n)

    if not spikes:
        return {
            "success": True,
            "elapsed_sec": round(time.time() - t0, 1),
            "results": [],
            "summary": f"No major chat spikes detected during {listen_sec}s window on {meta['title']}.",
        }

    results = []
    for idx, sp in enumerate(spikes, 1):
        if _is_already_clipped(meta["video_id"], sp["bucket_sec"]):
            logger.info(f"Skipping already-clipped moment on {meta['video_id']} (bucket {sp['bucket_sec']}).")
            continue

        raw_slice = WORKSPACE_DIR / f"live_slice_{idx}.mp4"
        seconds_ago = max(15.0, time.time() - sp["bucket_sec"])
        slice_live_segment(meta["webpage_url"], sp["start_ts"], sp["end_ts"], raw_slice, seconds_ago=seconds_ago)

        clip_start_sec, clip_end_sec = _align_to_audio_peak(raw_slice, total_sec=50.0)
        live_cues = _transcribe_live_cues(raw_slice)
        clip_meta = _generate_live_clip_meta(sp, meta)

        out_mp4 = OUTPUT_DIR / f"live_clip_{idx:02d}_{meta['video_id']}.mp4"
        preset = CATEGORY_TO_PRESET.get(clip_meta.get("category", "comedy"), "default")
        rendered = render_vertical_short(
            raw_path=raw_slice,
            out_path=out_mp4,
            hook_text=clip_meta.get("hook_overlay", "CHAT WENT CRAZY HERE"),
            cues=live_cues,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            preset=preset,
        )

        if raw_slice.exists():
            raw_slice.unlink()

        qc = check_video(out_mp4) if out_mp4.exists() else {"passed": False}
        if rendered and out_mp4.exists():
            _mark_clipped(meta["video_id"], sp["bucket_sec"])
        out_mp4.with_suffix(".json").write_text(json.dumps(clip_meta, indent=2), encoding="utf-8")
        results.append({
            "index": idx,
            "success": bool(rendered and out_mp4.exists()),
            "path": str(out_mp4),
            "clip": clip_meta,
            "qc": qc,
        })

    from drive_uploader import upload_clip_to_drive
    for r in results:
        if r.get("success"):
            r["drive"] = upload_clip_to_drive(Path(r["path"]))

    return {
        "success": True,
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
        "summary": f"Processed {len(results)} live chat spike(s) from {meta['title']}.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Live YouTube URL or @Channel/live")
    parser.add_argument("--listen", type=int, default=480, help="Seconds to monitor chat (default 480s)")
    parser.add_argument("--top", type=int, default=2, help="Max spike clips to render")
    args = parser.parse_args()
    print(json.dumps(run_live_monitor(args.url, listen_sec=args.listen, top_n=args.top), indent=2))