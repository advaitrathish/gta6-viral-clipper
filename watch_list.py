"""
watch_list.py — Multi-Creator GTA 6 Launch Radar (Hindi + English)
Automatically checks which creators are currently live, captures hype spikes,
renders vertical Shorts, and uploads them directly to Google Drive.
"""
import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from config import BASE_DIR
from live_chat_watcher import run_live_monitor
from drive_uploader import upload_clip_to_drive

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
    if COOKIES_FILE.exists():
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


def scan_and_clip(listen_sec: int = 360, top_per_creator: int = 1, max_creators: int = 3, upload_drive: bool = True) -> dict:
    """Scan all 8 creators in priority order, clip top spikes, and upload to Drive."""
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
            "summary": "None of the 8 watched creators are live right now.",
        }

    targets = active_streams[:max_creators]
    
    all_results = []
    for stream in targets:
        logger.info(f"Monitoring {stream['name']} for {listen_sec}s...")
        run_res = run_live_monitor(stream["webpage_url"], listen_sec=listen_sec, top_n=top_per_creator)
        for item in run_res.get("results", []):
            if item.get("success") and upload_drive:
                try:
                    drive_info = upload_clip_to_drive(Path(item["path"]))
                    item["drive"] = drive_info
                except Exception as e:
                    logger.warning(f"Drive upload failed for {item['path']}: {e}")
                    item["drive_error"] = str(e)
            all_results.append({"creator": stream["name"], **item})

    return {
        "success": True,
        "elapsed_sec": round(time.time() - t0, 1),
        "active_creators": [s["name"] for s in active_streams],
        "clips_created": len(all_results),
        "results": all_results,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=int, default=360, help="Seconds to listen per live creator (default 360s)")
    parser.add_argument("--top", type=int, default=1, help="Max clips per live creator (default 1)")
    parser.add_argument("--max-creators", type=int, default=3, help="Max live creators to process per run")
    parser.add_argument("--no-drive", action="store_true", help="Skip Google Drive upload")
    args = parser.parse_args()
    print(json.dumps(
        scan_and_clip(
            listen_sec=args.listen,
            top_per_creator=args.top,
            max_creators=args.max_creators,
            upload_drive=not args.no_drive,
        ),
        indent=2,
    ))