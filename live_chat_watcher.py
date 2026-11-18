"""
live_chat_watcher.py — Unified Live Stream, Ended Stream Replay (VOD), and Upload Clipper
Supports:
- mode="live": Real-time live chat spike listener
- mode="vod_replay": Hybrid Heatmap + Filtered Live Chat Replay analyzer for ended streams
- mode="upload": Standard uploaded video pipeline (heatmap/transcript via pipeline.py)
- mode="auto": Automatically detects live vs ended stream vs standard upload
"""
import argparse
import json
import logging
import re
import statistics
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

DEFAULT_MIN_VIRALITY_SCORE = 80
_WHISPER_MODEL = None

logger = logging.getLogger("live_chat_watcher")

HYPE_PATTERN = re.compile(
    r"\b(w+|l+|lmao+|lmfao+|lol+|wtf+|omg+|no\s*way|what+|bro+|clip(\s*it)?|gg+|ez+|"
    r"op+|bhai+|arey+|abe+|kya+|pagal|gajab|mast|khatarnak|haha+|xd+|💀|😭|🔥|😱|😂)\b",
    re.IGNORECASE,
)

SPAM_IGNORE_PATTERN = re.compile(
    r"\b(giveaway|redeem|code|superchat|member|membership|sub|subscribe|like\s*karo|"
    r"discord|instagram|insta|sponsor|ad|break|starting\s*soon|intro|shoutout)\b",
    re.IGNORECASE,
)


def _get_whisper_model():
    """Load faster-whisper once per runner execution to avoid repeated disk/CPU init."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel
        _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def _format_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _is_already_clipped(video_id: str, bucket_sec: float) -> bool:
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
    """Fetch video_id, title, is_live, was_live, duration, and release_timestamp from YouTube."""
    cmd = [
        "yt-dlp", "--dump-json", "--no-playlist", "--no-warnings",
        "--js-runtimes", "deno", "--remote-components", "ejs:github",
    ]
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 0:
        cmd += ["--cookies", str(COOKIES_FILE)]
    cmd.append(url)

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"yt-dlp metadata failed: {res.stderr.strip()[-600:]}")
    data = json.loads(res.stdout.splitlines()[0])
    return {
        "video_id": data.get("id"),
        "title": data.get("title", "Stream"),
        "channel": data.get("uploader", "Creator"),
        "is_live": bool(data.get("is_live")),
        "was_live": bool(data.get("was_live")),
        "duration": float(data.get("duration") or 0),
        "release_timestamp": data.get("release_timestamp"),
        "webpage_url": data.get("webpage_url", url),
    }


def collect_chat_json(url: str, duration_sec: int = 1800, is_vod_replay: bool = False) -> Path:
    """Download replay chat (VOD) up to 360s or record live chat for duration_sec without crashing."""
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
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 0:
        cmd += ["--cookies", str(COOKIES_FILE)]
    cmd.append(url)

    wait_limit = 360 if is_vod_replay else duration_sec
    logger.info(
        f"{'Downloading VOD Live Chat Replay JSON' if is_vod_replay else 'Listening to live chat'} "
        f"(max {wait_limit}s)..."
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        proc.wait(timeout=wait_limit)
    except subprocess.TimeoutExpired:
        logger.info("Chat capture window reached — stopping yt-dlp cleanly and using captured chat.")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    files = [
        p for p in WORKSPACE_DIR.glob("live_chat_capture*.live_chat.json*")
        if p.is_file() and p.stat().st_size > 0
    ]
    if not files:
        raise RuntimeError("No live_chat.json captured (chat may be disabled, huge, or still processing).")
    return files[0]


def detect_chat_spikes(
    chat_file: Path,
    stream_start_ts: float | None,
    top_k: int = 5,
    is_vod_replay: bool = False,
) -> list[dict]:
    """
    Parse live or replay chat JSONL and rank 10s buckets using relative surge + hype density.
    Filters out stream intros (< 4 min) and giveaway/promo chat spam while keeping a safe fallback.
    """
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

            if not isinstance(obj, dict):
                continue

            replay_offset_sec = None
            replay_action = obj.get("replayChatItemAction")
            if isinstance(replay_action, dict):
                actions = replay_action.get("actions") or []
                offset_ms = replay_action.get("videoOffsetTimeMsec")
                if offset_ms is not None:
                    try:
                        replay_offset_sec = max(0.0, float(offset_ms) / 1000.0)
                    except (TypeError, ValueError):
                        replay_offset_sec = None
            elif "addChatItemAction" in obj:
                actions = [obj]
            else:
                actions = []

            if isinstance(actions, dict):
                actions = [actions]
            if not isinstance(actions, list):
                continue

            for act in actions:
                if not isinstance(act, dict):
                    continue
                add_action = act.get("addChatItemAction")
                if not isinstance(add_action, dict):
                    continue
                item = add_action.get("item")
                if not isinstance(item, dict):
                    continue

                renderer = (
                    item.get("liveChatTextMessageRenderer")
                    or item.get("liveChatPaidMessageRenderer")
                )
                if not isinstance(renderer, dict):
                    continue

                msg_sec = None
                if is_vod_replay and replay_offset_sec is not None:
                    msg_sec = replay_offset_sec
                else:
                    usec = renderer.get("timestampUsec")
                    if usec:
                        try:
                            msg_sec = int(usec) / 1_000_000.0
                        except (TypeError, ValueError):
                            msg_sec = None
                    if msg_sec is None and replay_offset_sec is not None:
                        msg_sec = replay_offset_sec

                if msg_sec is None:
                    continue

                if first_msg_sec is None:
                    first_msg_sec = msg_sec

                author_name_obj = renderer.get("authorName")
                fallback_author = (
                    author_name_obj.get("simpleText", "anon")
                    if isinstance(author_name_obj, dict)
                    else "anon"
                )
                author = renderer.get("authorExternalChannelId") or fallback_author

                msg_obj = renderer.get("message")
                runs = msg_obj.get("runs", []) if isinstance(msg_obj, dict) else []
                if not isinstance(runs, list):
                    continue

                text = "".join(r.get("text", "") for r in runs if isinstance(r, dict)).strip()
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

    totals = [len(v["authors"]) for v in buckets.values()]
    hypes = [len(v["hype_authors"]) for v in buckets.values()]
    median_total = max(3.0, float(statistics.median(totals)))
    median_hype = max(2.0, float(statistics.median(hypes)))

    base_ts = 0.0 if is_vod_replay else (float(stream_start_ts) if stream_start_ts else float(first_msg_sec or 0))
    clean_scored = []
    unfiltered_fallback = []

    for b_sec, info in sorted(buckets.items()):
        unique_total = len(info["authors"])
        unique_hype = len(info["hype_authors"])
        if unique_total < 4 and unique_hype < 2:
            continue

        surge_ratio = (unique_total / median_total) + (unique_hype / median_hype) * 1.5
        hype_density = unique_hype / max(1, unique_total)
        score = (unique_hype * 2.0) + (unique_total * 0.6) + (surge_ratio * 12.0) + (hype_density * 15.0)
        offset_sec = float(b_sec) if is_vod_replay else max(0.0, b_sec - base_ts)

        entry = {
            "bucket_sec": b_sec,
            "offset_sec": offset_sec,
            "score": round(score, 2),
            "unique_total": unique_total,
            "unique_hype": unique_hype,
            "sample_chat": info["messages"],
        }
        unfiltered_fallback.append(entry)

        # Filter 1: Skip stream starting/intro screen (first 4 minutes of VOD)
        if is_vod_replay and offset_sec < 240.0:
            continue

        # Filter 2: Skip giveaway/promo/shoutout chat spam floods
        spam_count = sum(1 for m in info["messages"] if SPAM_IGNORE_PATTERN.search(m))
        if spam_count >= max(3, int(len(info["messages"]) * 0.4)):
            continue

        clean_scored.append(entry)

    # Guarantee we never end up empty if filters were too strict on a short stream
    pool = clean_scored if clean_scored else unfiltered_fallback
    pool.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for cand in pool:
        if any(abs(cand["offset_sec"] - s["offset_sec"]) < 75 for s in selected):
            continue
        start_s = max(0.0, cand["offset_sec"] - 38.0)
        end_s = start_s + 50.0
        cand["start_ts"] = _format_ts(start_s)
        cand["end_ts"] = _format_ts(end_s)
        cand["start_sec"] = start_s
        cand["end_sec"] = end_s
        selected.append(cand)
        if len(selected) >= top_k:
            break

    return selected


def slice_vod_segment(url: str, start_sec: float, end_sec: float, out_path: Path) -> Path:
    """Download ONLY the 50s highlight window from an ended stream/VOD using yt-dlp --download-sections."""
    if out_path.exists():
        out_path.unlink()

    section_spec = f"*{_format_ts(start_sec)}-{_format_ts(end_sec)}"
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "--download-sections", section_spec,
        "--force-keyframes-at-cuts",
        "--js-runtimes", "deno",
        "--remote-components", "ejs:github",
        "-o", str(out_path),
    ]
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 0:
        cmd += ["--cookies", str(COOKIES_FILE)]
    cmd.append(url)

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"VOD section slice failed ({section_spec}): {res.stderr.strip()[-400:]}")
    return out_path


def slice_live_segment(url: str, start_ts: str, end_ts: str, out_path: Path, seconds_ago: float = 60.0) -> Path:
    """Resolve HLS m3u8 live URL via yt-dlp -g and slice 50s from the live DVR buffer with FFmpeg."""
    if out_path.exists():
        out_path.unlink()

    url_cmd = [
        "yt-dlp",
        "-g",
        "-f", "best[protocol*=m3u8][height<=1080]/best[height<=1080][protocol!*=dash]/best",
        "--remote-components", "ejs:github",
    ]
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 0:
        url_cmd += ["--cookies", str(COOKIES_FILE)]
    url_cmd.append(url)

    u_res = subprocess.run(url_cmd, capture_output=True, text=True, timeout=60)
    stream_urls = [line.strip() for line in u_res.stdout.splitlines() if line.strip().startswith("http")]
    if u_res.returncode != 0 or not stream_urls:
        raise RuntimeError(f"Failed to resolve HLS live URL: {u_res.stderr.strip()[:300]}")

    ffmpeg_bin = get_ffmpeg_exe()
    seek_back = f"-{max(55, int(seconds_ago + 38))}"

    if len(stream_urls) == 1:
        inputs = ["-sseof", seek_back, "-i", stream_urls[0]]
        maps = ["-map", "0:v:0", "-map", "0:a?"]
    else:
        inputs = ["-sseof", seek_back, "-i", stream_urls[0], "-sseof", seek_back, "-i", stream_urls[1]]
        maps = ["-map", "0:v:0", "-map", "1:a:0"]

    ff_cmd = [
        ffmpeg_bin, "-y",
        *inputs,
        *maps,
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
    """Find the loudest streamer reaction peak in the slice and frame [peak-8s, peak+32s]."""
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
            start_sec = max(0.0, round(peak_t - 8.0, 1))
            end_sec = min(total_sec, round(start_sec + 40.0, 1))
            if end_sec - start_sec >= 20.0:
                logger.info(f"Audio peak aligned: {start_sec}s -> {end_sec}s (peak @ {peak_t:.1f}s)")
                return start_sec, end_sec
    except Exception as e:
        logger.warning(f"Audio alignment skipped: {e}")
    return 0.0, total_sec


def _transcribe_live_cues(raw_path: Path) -> list[dict]:
    """Generate timestamped caption cues from audio using cached faster-whisper model."""
    try:
        model = _get_whisper_model()
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
        logger.info(f"Transcribed {len(cues)} caption segments via faster-whisper.")
        return cues
    except Exception as e:
        logger.warning(f"Caption transcription skipped: {e}")
        return []


def _default_spike_meta(spike: dict, stream_meta: dict) -> dict:
    return {
        "start_time": "00:00:00",
        "end_time": "00:00:50",
        "duration_sec": 50.0,
        "virality_score": min(99, int(76 + (spike["unique_hype"] * 1.2))),
        "confidence": 88,
        "category": "comedy",
        "title": f"{stream_meta['channel']} Insane Stream Moment",
        "hook_overlay": "LIVE CHAT EXPLODED AT THIS MOMENT",
        "mid_callout": "WATCH THE STREAMER REACTION HERE",
        "hashtags": ["#Shorts", "#GTA6", "#Gaming", "#Live"],
        "reason": f"Chat spiked with {spike['unique_hype']} unique viewers reacting simultaneously.",
    }


def _score_spikes_batched(spikes: list[dict], stream_meta: dict, max_clips: int) -> list[tuple[int, dict, dict]]:
    """Score all top chat spikes in ONE single Gemini request so fallback takes < 5 seconds."""
    if not spikes:
        return []

    combined_lines = []
    chat_spike_items = []
    for sp in spikes:
        s_sec = sp["start_sec"]
        e_sec = sp["end_sec"]
        sample_txt = " | ".join(sp["sample_chat"][:12])
        combined_lines.append(
            f"[{s_sec:.1f}s] WINDOW {s_sec:.1f}s-{e_sec:.1f}s | CHAT SPIKE "
            f"({sp['unique_hype']} hype / {sp['unique_total']} active): {sample_txt}"
        )
        chat_spike_items.append({
            "start_sec": sp["offset_sec"],
            "msgs_in_10s": sp["unique_total"],
            "hype_msgs": sp["unique_hype"],
        })

    try:
        signals = {
            "video_id": stream_meta.get("video_id", "live"),
            "title": stream_meta.get("title", "Stream"),
            "duration_sec": float(stream_meta.get("duration") or 0.0),
            "transcript": "\n".join(combined_lines),
            "heatmap_peaks": [],
            "chat_spikes": chat_spike_items,
        }
        ai_clips = find_viral_clips(signals=signals, max_clips=max(max_clips, len(spikes)))
        if ai_clips:
            matched: list[tuple[int, dict, dict]] = []
            used_spikes = set()
            for c in ai_clips:
                c_mid = (float(c["start_sec"]) + float(c["end_sec"])) / 2.0
                best_idx = None
                best_dist = 999999.0
                for idx, sp in enumerate(spikes):
                    if idx in used_spikes:
                        continue
                    dist = abs(sp["offset_sec"] - c_mid)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                if best_idx is not None and best_dist <= 90.0:
                    used_spikes.add(best_idx)
                    sp = spikes[best_idx]
                    c_copy = dict(c)
                    c_copy["start_time"] = "00:00:00"
                    c_copy["end_time"] = "00:00:50"
                    c_copy["duration_sec"] = 50.0
                    matched.append((int(c_copy.get("virality_score", 85)), sp, c_copy))

            for idx, sp in enumerate(spikes):
                if idx not in used_spikes:
                    fb = _default_spike_meta(sp, stream_meta)
                    matched.append((fb["virality_score"], sp, fb))
            return matched
    except Exception as e:
        logger.warning(f"Gemini batched metadata fallback triggered: {e}")

    return [(_default_spike_meta(sp, stream_meta)["virality_score"], sp, _default_spike_meta(sp, stream_meta)) for sp in spikes]


def run_upload_pipeline(url: str, min_clips: int = 1, max_clips: int = 2) -> dict:
    """Standard uploaded video / heatmap mode (uses heatmap + transcript via pipeline.py)."""
    from pipeline import run_pipeline
    from drive_uploader import upload_clip_to_drive

    target_n = max(min_clips, max_clips)
    vod_res = run_pipeline(url=url, max_clips=max(5, target_n + 2), top_n=target_n)
    for r in vod_res.get("results", []):
        if r.get("success") and r.get("path"):
            r["drive"] = upload_clip_to_drive(Path(r["path"]))
    return vod_res


def run_monitor(
    url: str,
    mode: str = "auto",
    listen_sec: int = 1800,
    min_clips: int = 1,
    max_clips: int = 2,
) -> dict:
    """Unified entrypoint for live streams, ended stream chat/heatmap replays, and uploaded videos."""
    t0 = time.time()
    max_clips = max(1, max_clips)
    min_clips = max(0, min(min_clips, max_clips))

    meta = get_live_metadata(url)

    if mode == "auto":
        if meta["is_live"]:
            effective_mode = "live"
        elif meta["was_live"]:
            effective_mode = "vod_replay"
        else:
            effective_mode = "upload"
    else:
        effective_mode = mode

    logger.info(f"Running mode='{effective_mode}' on '{meta['title']}' (min={min_clips}, max={max_clips})")

    if effective_mode == "upload":
        return run_upload_pipeline(meta["webpage_url"], min_clips=min_clips, max_clips=max_clips)

    is_vod_replay = (effective_mode == "vod_replay")
    if effective_mode == "live" and not meta["is_live"]:
        logger.warning("Stream is no longer live — automatically switching to 'vod_replay' mode!")
        is_vod_replay = True

    # For ended streams (vod_replay), check YouTube's 'Most Replayed' heatmap first:
    # viewers rewind real gameplay moments, skipping intro screens & giveaway chat floods.
    if is_vod_replay:
        try:
            from extractor import fetch_video_signals
            sigs = fetch_video_signals(meta["webpage_url"])
            if sigs.get("heatmap_peaks"):
                logger.info(
                    f"Found {len(sigs['heatmap_peaks'])} YouTube 'Most Replayed' heatmap peaks — "
                    "running heatmap + transcript pipeline!"
                )
                return run_upload_pipeline(meta["webpage_url"], min_clips=min_clips, max_clips=max_clips)
        except Exception as e:
            logger.info(f"Heatmap check skipped ({e}) — falling back to filtered VOD chat replay.")

    try:
        chat_file = collect_chat_json(
            meta["webpage_url"],
            duration_sec=listen_sec,
            is_vod_replay=is_vod_replay,
        )
    except RuntimeError as e:
        if is_vod_replay:
            logger.warning(f"Replay chat unavailable ({e}) — falling back to upload heatmap pipeline.")
            return run_upload_pipeline(meta["webpage_url"], min_clips=min_clips, max_clips=max_clips)
        raise

    candidate_pool = max(4, max_clips * 2)
    spikes = detect_chat_spikes(
        chat_file,
        meta.get("release_timestamp"),
        top_k=candidate_pool,
        is_vod_replay=is_vod_replay,
    )

    if not spikes:
        if is_vod_replay:
            logger.warning("No chat replay spikes found — falling back to upload heatmap pipeline.")
            return run_upload_pipeline(meta["webpage_url"], min_clips=min_clips, max_clips=max_clips)
        return {
            "success": True,
            "elapsed_sec": round(time.time() - t0, 1),
            "results": [],
            "summary": f"No major chat spikes detected during {listen_sec}s window on {meta['title']} — skipped.",
        }

    unclipped_spikes = [
        sp for sp in spikes
        if is_vod_replay or not _is_already_clipped(meta["video_id"], sp["bucket_sec"])
    ]

    scored_candidates = _score_spikes_batched(unclipped_spikes, meta, max_clips=max_clips)
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    chosen = []
    for v_score, sp, clip_meta in scored_candidates:
        if len(chosen) >= max_clips:
            break
        if v_score >= DEFAULT_MIN_VIRALITY_SCORE or len(chosen) < min_clips:
            chosen.append((v_score, sp, clip_meta))

    results = []
    for idx, (v_score, sp, clip_meta) in enumerate(chosen, 1):
        raw_slice = WORKSPACE_DIR / f"clip_slice_{idx}.mp4"
        if is_vod_replay:
            slice_vod_segment(meta["webpage_url"], sp["start_sec"], sp["end_sec"], raw_slice)
        else:
            seconds_ago = max(15.0, time.time() - sp["bucket_sec"])
            slice_live_segment(meta["webpage_url"], sp["start_ts"], sp["end_ts"], raw_slice, seconds_ago=seconds_ago)

        clip_start_sec, clip_end_sec = _align_to_audio_peak(raw_slice, total_sec=50.0)
        cues = _transcribe_live_cues(raw_slice)

        prefix = "vod" if is_vod_replay else "live"
        out_mp4 = OUTPUT_DIR / f"{prefix}_clip_{idx:02d}_{meta['video_id']}.mp4"
        preset = CATEGORY_TO_PRESET.get(clip_meta.get("category", "comedy"), "default")
        rendered = render_vertical_short(
            raw_path=raw_slice,
            out_path=out_mp4,
            hook_text=clip_meta.get("hook_overlay", "LIVE CHAT EXPLODED AT THIS MOMENT"),
            cues=cues,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            preset=preset,
            mid_text=clip_meta.get("mid_callout", ""),
        )

        if raw_slice.exists():
            raw_slice.unlink()

        qc = check_video(out_mp4) if out_mp4.exists() else {"passed": False}
        if rendered and out_mp4.exists():
            _mark_clipped(meta["video_id"], sp["bucket_sec"])
            out_mp4.with_suffix(".json").write_text(json.dumps(clip_meta, indent=2), encoding="utf-8")
            results.append({
                "index": idx,
                "success": True,
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
        "summary": f"Processed {len(results)} clip(s) [{effective_mode}] from {meta['title']}.",
    }


def run_live_monitor(url: str, listen_sec: int = 1800, top_n: int = 1) -> dict:
    """Backward-compatible wrapper for watch_list.py live radar."""
    return run_monitor(url=url, mode="live", listen_sec=listen_sec, min_clips=0, max_clips=top_n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube URL (Live, Ended Stream VOD, or Upload)")
    parser.add_argument(
        "--mode",
        choices=["auto", "live", "vod_replay", "upload"],
        default="auto",
        help="Clipping mode (default: auto)",
    )
    parser.add_argument("--listen", type=int, default=1800, help="Seconds to monitor live chat (live mode only)")
    parser.add_argument("--min-clips", type=int, default=1, help="Minimum clips to guarantee (default 1)")
    parser.add_argument("--max-clips", type=int, default=2, help="Maximum clips to render (default 2)")
    parser.add_argument("--top", type=int, default=None, help="Legacy alias for --max-clips")
    args = parser.parse_args()

    max_c = args.top if args.top is not None else args.max_clips
    print(
        json.dumps(
            run_monitor(
                args.url,
                mode=args.mode,
                listen_sec=args.listen,
                min_clips=args.min_clips,
                max_clips=max_c,
            ),
            indent=2,
        )
    )