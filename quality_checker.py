"""
quality_checker.py — Output validation for GTA6 AI Content Factory
Every rendered clip is checked before being accepted for review or upload.
"""
import json
import re
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional
import cv2


def _probe_video_info(path: Path) -> Optional[Dict]:
    """Probes resolution, duration, and audio stream via ffprobe or OpenCV+ffmpeg fallback."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    "-show_format",
                    str(path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
                a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
                if v_stream:
                    return {
                        "width": int(v_stream.get("width", 0)),
                        "height": int(v_stream.get("height", 0)),
                        "duration": float(data.get("format", {}).get("duration", 0) or 0),
                        "has_audio": a_stream is not None,
                    }
        except Exception:
            pass

    # Fallback when ffprobe is not on PATH (e.g., imageio-ffmpeg only)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if width <= 0 or height <= 0:
        return None

    duration = round(frames / fps, 2) if fps > 0 else 0.0
    has_audio = True
    try:
        from clipper import get_ffmpeg_exe
        res = subprocess.run(
            [get_ffmpeg_exe(), "-i", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        has_audio = bool(re.search(r"Stream #\d+:\d+.*Audio:", res.stderr))
    except Exception:
        pass

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "has_audio": has_audio,
    }


def check_video(path: Path, min_duration: float = 20.0, max_duration: float = 65.0) -> Dict:
    result = {
        "ok": False,
        "path": str(path),
        "issues": [],
        "width": 0,
        "height": 0,
        "duration": 0.0,
        "has_audio": False,
    }

    if not path.exists():
        result["issues"].append("File does not exist")
        return result

    if not path.is_file():
        result["issues"].append("Path is not a file")
        return result

    size = path.stat().st_size
    if size < 50_000:
        result["issues"].append(f"File too small ({size} bytes) — likely corrupt or empty")
        return result

    info = _probe_video_info(path)
    if not info:
        result["issues"].append("Video probe failed — file may be unreadable or corrupt")
        return result

    width = info["width"]
    height = info["height"]
    result["width"] = width
    result["height"] = height

    if width != 1080 or height != 1920:
        result["issues"].append(f"Wrong resolution: {width}x{height} (expected 1080x1920)")

    if width > 0 and height > 0:
        ratio = height / width
        if abs(ratio - (16 / 9)) > 0.05:
            result["issues"].append(
                f"Aspect ratio {width}:{height} = {ratio:.3f} (expected ~1.778 for 9:16)"
            )

    fmt_duration = info["duration"]
    result["duration"] = round(fmt_duration, 2)

    if fmt_duration < min_duration:
        result["issues"].append(
            f"Duration too short: {fmt_duration:.1f}s (min {min_duration}s)"
        )
    if fmt_duration > max_duration:
        result["issues"].append(
            f"Duration too long: {fmt_duration:.1f}s (max {max_duration}s)"
        )

    result["has_audio"] = info["has_audio"]
    if not info["has_audio"]:
        result["issues"].append("No audio stream — verify this is intentional")

    critical_keywords = [
        "File does not exist",
        "File too small",
        "Video probe failed",
        "Wrong resolution",
        "Duration too short",
        "Duration too long",
    ]
    hard_fails = [
        issue for issue in result["issues"]
        if any(kw in issue for kw in critical_keywords)
    ]
    result["ok"] = len(hard_fails) == 0

    return result


def check_clips(paths: list, min_duration: float = 20.0, max_duration: float = 65.0) -> Dict:
    passed = []
    failed = []

    for p in paths:
        r = check_video(Path(p), min_duration, max_duration)
        if r["ok"]:
            passed.append(r)
            print(f"[QC] PASS  {Path(p).name}  ({r['width']}x{r['height']}, {r['duration']:.1f}s)")
        else:
            failed.append(r)
            print(f"[QC] FAIL  {Path(p).name}")
            for issue in r["issues"]:
                print(f"       -> {issue}")

    return {"passed": passed, "failed": failed, "total": len(paths)}


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(Path("output_clips").glob("*.mp4"))
    if not targets:
        print("Usage: python quality_checker.py [video.mp4 ...]")
    else:
        summary = check_clips([str(t) for t in targets])
        print(f"\n=== QC SUMMARY: {len(summary['passed'])}/{summary['total']} passed ===")