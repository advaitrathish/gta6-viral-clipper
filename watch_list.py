"""
watch_list.py — Multi-Creator GTA 6 Launch Radar (Hindi + English)
Scans all 8 creators, listens for up to 30 minutes total across active streams,
extracts only the single highest-virality Short (if one exists), and uploads to Drive.
"""
import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from config import BASE_DIR
from live_chat_watcher import run_live_monitor

COOKIES_FILE = BASE_DIR / "cookies.txt"
logger = logging.getLogger("watch_list")

CREATORS = [
    {"name": "IShowSpeed", "handle": "@IShowSpeed", "lang": "en"},
    {"name": "CarryMinati", "handle": "@CarryMinati", "lang": "hi"},
    {"name": "Techno Gamerz", "handle": "@TechnoGamerzOfficial", "lang": "hi"},
    {"name": "Triggered Insaan", "handle": "@TriggeredInsaan", "lang": "hi"},
    {"name": "Total Gaming", "handle": "@TotalGaming093", "lang": "hi"},
    {"name": "Payal Gaming", "handle": "@PayalGaming", "lang": "hi"},
    {"name": "Kai Cenat", "handle": "@KaiCenat", "lang": "en"},
    {"name": "Ludwig", "handle": "@ludwig", "lang": "en"},
]


def check_channel_live(handle: str) -> dict | None:
    """Check if `https://www.youtube.com/<handle>/live` is actively streaming right now."""
    live_url = f"https://www.youtube.com/{handle}/live"
    cmd = ["yt-dlp", "--dump-json", "--no-playlist", "--remote-components", "ejs:github"]
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 0:
        cmd += ["--cookies", str(COOKIES_FILE)]
    cmd.append(live_url)

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0 or not res.stdout.strip():
            return None
        data = json.loads(res.stdout.splitlines()[0])
        if data.get("is_live"):
            return {
                "video_id": data.get("id"),
                "title": data.get("title", ""),
                "webpage_url": data.get("webpage_url", live_url),
                "concurrent_viewers": data.get("concurrent_view_count") or 0,
            }
    except Exception:
        pass
    return None


def scan_and_clip(listen_sec: int = 1800, top_total: int = 1, max_creators: int = 2) -> dict:
    """Scan all 8 creators, split the 30-min budget across active streams, and keep only top_total viral clips."""
    t0 = time.time()
    active_streams = []

    for c in CREATORS:
        logger.info(f"Checking {c['name']} ({c['handle']})...")
        info = check_channel_live(c["handle"])
        if info:
            logger.info(f"LIVE FOUND: {c['name']} -> {info['title']} ({info['concurrent_viewers']} viewers)")
            active_streams.append({**c, **info})

    if not active_streams:
        return {
            "success": True,
            "elapsed_sec": round(time.time() - t0, 1),
            "active_creators": [],
            "clips_created": 0,
            "summary": "None of the 8 watched creators are live right now — skipping run.",
        }

    targets = active_streams[:max_creators]
    per_creator_listen = max(300, listen_sec // len(targets))
    all_results = []

    for stream in targets:
        if len(all_results) >= top_total:
            logger.info("Target viral clip already captured for this hourly window — stopping radar.")
            break

        remaining_clips = top_total - len(all_results)
        logger.info(f"Monitoring {stream['name']} for {per_creator_listen}s (seeking top {remaining_clips} viral clip)...")
        run_res = run_live_monitor(
            stream["webpage_url"],
            listen_sec=per_creator_listen,
            top_n=remaining_clips,
        )
        for item in run_res.get("results", []):
            if item.get("success"):
                all_results.append({"creator": stream["name"], **item})

    return {
        "success": True,
        "elapsed_sec": round(time.time() - t0, 1),
        "active_creators": [s["name"] for s in active_streams],
        "clips_created": len(all_results),
        "results": all_results,
        "summary": (
            f"Captured {len(all_results)} viral clip(s) from {len(active_streams)} live creator(s)."
            if all_results
            else "No moment met the virality threshold during this 30-minute window — skipped."
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=int, default=1800, help="Total seconds to monitor live chat (default 1800s = 30m)")
    parser.add_argument("--top", type=int, default=1, help="Max total viral clips per hourly run (default 1)")
    parser.add_argument("--max-creators", type=int, default=2, help="Max live creators to split the 30m window across")
    args = parser.parse_args()
    print(json.dumps(
        scan_and_clip(
            listen_sec=args.listen,
            top_total=args.top,
            max_creators=args.max_creators,
        ),
        indent=2,
    ))