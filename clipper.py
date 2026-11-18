"""
clipper.py — Video download, rendering, and editing for GTA6 AI Content Factory
Renders 1080x1920 (9:16) vertical Shorts with:
- Full corner webcam capture on top + gameplay on bottom (no head/hair cutoff)
- Four editing presets: comedy, action, cinematic, discovery (optimized with veryfast preset)
- ASS caption burning
- Audio loudness normalization
- Hook text overlay via textfile (safe for all Unicode/special chars)
"""
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from config import (
    WORK_DIR, OUTPUT_DIR, FONT_PATH,
    AUDIO_LOUDNESS_TARGET, AUDIO_TRUE_PEAK_TARGET,
)
from extractor import get_yt_dlp_base, seconds_to_hms
from caption_engine import generate_ass_captions

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = WORK_DIR / "face_detection_yunet_2023mar.onnx"

PRESET_STYLES = {
    "comedy": {
        "crf": "22",
        "preset": "veryfast",
        "saturation": 1.1,
        "contrast": 1.05,
    },
    "action": {
        "crf": "21",
        "preset": "veryfast",
        "saturation": 1.2,
        "contrast": 1.1,
    },
    "cinematic": {
        "crf": "20",
        "preset": "veryfast",
        "saturation": 0.95,
        "contrast": 1.0,
    },
    "discovery": {
        "crf": "22",
        "preset": "veryfast",
        "saturation": 1.05,
        "contrast": 1.05,
    },
    "default": {
        "crf": "22",
        "preset": "veryfast",
        "saturation": 1.0,
        "contrast": 1.0,
    },
}

CATEGORY_TO_PRESET = {
    "comedy": "comedy",
    "fail": "comedy",
    "unexpected": "comedy",
    "action": "action",
    "cinematic": "cinematic",
    "discovery": "discovery",
    "story": "discovery",
}


def get_ffmpeg_exe() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return "ffmpeg"


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]+', "", name).strip()
    return cleaned[:55] or "viral_clip"


def _escape_ffmpeg_filter_path(p: Path) -> str:
    """Escapes Windows drive-letter colons (C:/ -> C\\:/) for FFmpeg filter arguments."""
    return p.as_posix().replace(":", "\\:")


def _write_text_file(text: str, tmp_dir: Path) -> Path:
    p = tmp_dir / "hook_text.txt"
    p.write_text(text.upper(), encoding="utf-8")
    return p


def get_yunet_detector(w: int, h: int):
    if not YUNET_PATH.exists():
        print("[Clipper] Downloading YuNet face detection model...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (w, h), 0.70, 0.3, 5000)


def detect_facecam_corner(video_path: Path) -> Optional[str]:
    """Detects which screen corner holds the streamer webcam using YuNet."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        return None

    detector = get_yunet_detector(width, height)
    votes: Dict[str, int] = {
        "top-right": 0,
        "top-left": 0,
        "bottom-right": 0,
        "bottom-left": 0,
    }

    for p in (0.15, 0.35, 0.55, 0.75, 0.90):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * p))
        ok, frame = cap.read()
        if not ok:
            continue

        _, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            face = faces[0]
            cx = float(face[0]) + (float(face[2]) / 2.0)
            cy = float(face[1]) + (float(face[3]) / 2.0)
            is_right = cx >= (width * 0.5)
            is_bottom = cy >= (height * 0.5)

            if is_right and not is_bottom:
                votes["top-right"] += 1
            elif not is_right and not is_bottom:
                votes["top-left"] += 1
            elif is_right and is_bottom:
                votes["bottom-right"] += 1
            else:
                votes["bottom-left"] += 1

    cap.release()
    best_corner, count = max(votes.items(), key=lambda kv: kv[1])
    return best_corner if count > 0 else None


def download_clip_slice(url: str, start_hms: str, end_hms: str, raw_path: Path) -> bool:
    raw_path.unlink(missing_ok=True)
    section_arg = f"*{start_hms}-{end_hms}"
    ff_exe = get_ffmpeg_exe()
    ff_args = ["--ffmpeg-location", ff_exe] if ff_exe != "ffmpeg" else []

    cmd = get_yt_dlp_base() + ff_args + [
        "--download-sections", section_arg,
        "--force-keyframes-at-cuts",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(raw_path),
        url,
    ]
    print(f"[Clipper] Downloading slice {start_hms} -> {end_hms}...")
    res = subprocess.run(cmd, check=False)
    ok = res.returncode == 0 and raw_path.exists() and raw_path.stat().st_size > 10_000
    if not ok:
        print(f"[Clipper] Warning: Download failed for slice {start_hms} -> {end_hms}")
    return ok


def _build_eq_filter(sat: float, contrast: float) -> str:
    return f"eq=saturation={sat:.2f}:contrast={contrast:.2f}"


_NVENC_AVAILABLE: Optional[bool] = None


def _has_nvenc(ff_exe: str) -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        probe = subprocess.run(
            [
                ff_exe, "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                "-c:v", "h264_nvenc", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=10,
        )
        _NVENC_AVAILABLE = (probe.returncode == 0)
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def _detect_trim_end(ff_exe: str, raw_path: Path, min_duration: float = 18.0) -> Optional[float]:
    try:
        res = subprocess.run(
            [
                ff_exe, "-i", str(raw_path),
                "-af", "silencedetect=noise=-34dB:d=1.2",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=20,
        )
        starts = [float(m) for m in re.findall(r"silence_start:\s*([0-9.]+)", res.stderr)]
        ends = [float(m) for m in re.findall(r"silence_end:\s*([0-9.]+)", res.stderr)]
        if starts and len(starts) > len(ends) and starts[-1] >= min_duration:
            return round(starts[-1] + 0.35, 2)
    except Exception:
        pass
    return None


def render_vertical_short(
    raw_path: Path,
    out_path: Path,
    hook_text: str,
    cues: List[Dict],
    clip_start_sec: float,
    clip_end_sec: float,
    preset: str = "default",
) -> bool:
    ff_exe = get_ffmpeg_exe()
    style = PRESET_STYLES.get(preset, PRESET_STYLES["default"])
    eq_filt = _build_eq_filter(style["saturation"], style["contrast"])
    use_nvenc = _has_nvenc(ff_exe)
    trim_end = _detect_trim_end(ff_exe, raw_path)

    corner = detect_facecam_corner(raw_path)

    with tempfile.TemporaryDirectory(prefix="gta6_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        ass_path = tmp_path / "captions.ass"
        font_name = Path(FONT_PATH).stem if Path(FONT_PATH).exists() else "Liberation Sans"
        has_caps = generate_ass_captions(
            cues, ass_path, clip_start_sec, clip_end_sec, font_name, 72, preset
        )

        hook_file = _write_text_file(hook_text, tmp_path)
        hook_y = "915" if corner else "240"

        hook_file_escaped = _escape_ffmpeg_filter_path(hook_file)
        font_part = ""
        if Path(FONT_PATH).exists():
            font_escaped = _escape_ffmpeg_filter_path(Path(FONT_PATH))
            font_part = f"fontfile='{font_escaped}':"

        drawtext = (
            f"drawtext=textfile='{hook_file_escaped}':"
            f"{font_part}"
            f"fontcolor=white:fontsize=54:"
            f"box=1:boxcolor=black@0.80:boxborderw=16:"
            f"x=(w-text_w)/2:y={hook_y}:"
            f"fix_bounds=1"
        )

        ass_escaped = _escape_ffmpeg_filter_path(ass_path)
        ass_filter = f"ass='{ass_escaped}'" if has_caps else ""

        # Full 16:9 corner slice (36% width x 36% height) captures the entire webcam box
        corner_crops = {
            "top-right":    "crop=iw*0.36:ih*0.36:iw*0.64:0",
            "top-left":     "crop=iw*0.36:ih*0.36:0:0",
            "bottom-right": "crop=iw*0.36:ih*0.36:iw*0.64:ih*0.64",
            "bottom-left":  "crop=iw*0.36:ih*0.36:0:ih*0.64",
        }

        layout_label = f"Full Corner Webcam ({corner})" if corner else "Full-Frame Gameplay (No Facecam)"
        print(f"[Clipper] Layout -> {layout_label} | Preset -> {preset.upper()}")

        hook_zoom = (
            "scale=w='if(lt(t,3),iw*(1+0.08*(1-t/3)),iw)':"
            "h='if(lt(t,3),ih*(1+0.08*(1-t/3)),ih)':eval=frame,"
            "crop=1080:1920:(iw-1080)/2:(ih-1920)/2"
        )

        if corner:
            cam_crop = corner_crops[corner]
            # Fit entire webcam inside top 1080x960 pane without zooming/chopping top or bottom
            filter_parts = [
                f"[0:v]{cam_crop},scale=1080:960:force_original_aspect_ratio=increase,"
                f"crop=1080:960,boxblur=20:8,{eq_filt},eq=brightness=-0.15,setsar=1[cam_bg]",
                f"[0:v]{cam_crop},{eq_filt},scale=1080:960:force_original_aspect_ratio=decrease,setsar=1[cam_fg]",
                f"[cam_bg][cam_fg]overlay=(W-w)/2:(H-h)/2,setsar=1[cam]",
                f"[0:v]crop=ih*9/8:ih:(iw-ih*9/8)/2:0,{eq_filt},scale=1080:960,setsar=1[game]",
                f"[cam][game]vstack=inputs=2,{hook_zoom}[stacked]",
            ]
            last_label = "stacked"
        else:
            filter_parts = [
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,boxblur=25:10,{eq_filt},eq=brightness=-0.12[bg]",
                f"[0:v]{eq_filt},scale=1080:-2[fg]",
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{hook_zoom}[comp]",
            ]
            last_label = "comp"

        if ass_filter:
            filter_parts.append(f"[{last_label}]{ass_filter}[captioned]")
            last_label = "captioned"

        filter_parts.append(f"[{last_label}]{drawtext}[outv]")
        filter_complex = ";".join(filter_parts)

        audio_filter = (
            f"loudnorm=I={AUDIO_LOUDNESS_TARGET}:"
            f"TP={AUDIO_TRUE_PEAK_TARGET}:LRA=11"
        )

        vcodec_args = (
            ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(style["crf"]), "-rc", "vbr"]
            if use_nvenc
            else ["-c:v", "libx264", "-preset", style["preset"], "-crf", str(style["crf"])]
        )

        cmd = [ff_exe, "-y", "-i", str(raw_path)]
        if trim_end:
            cmd += ["-t", str(trim_end)]
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-af", audio_filter,
            *vcodec_args,
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]

        enc_label = "NVENC GPU" if use_nvenc else "CPU libx264 (veryfast)"
        print(f"[Clipper] Rendering {preset.upper()} ({enc_label}) -> {out_path.name}...")
        res = subprocess.run(cmd, check=False)

        if res.returncode != 0 or not out_path.exists():
            print("[Clipper] Warning: Primary render failed. Retrying with safe CPU fallback...")
            cmd_fallback = [
                ff_exe, "-y",
                "-i", str(raw_path),
                "-filter_complex", "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[outv]",
                "-map", "[outv]",
                "-map", "0:a?",
                "-af", audio_filter,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ]
            res = subprocess.run(cmd_fallback, check=False)

        return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 10_000


def build_clips(
    url: str,
    clips: List[Dict],
    cues: List[Dict],
    top_n: int = 3,
) -> List[Dict]:
    ranked = sorted(clips, key=lambda c: c.get("virality_score", 0), reverse=True)[:top_n]
    generated = []

    for idx, clip in enumerate(ranked, 1):
        raw_file = WORK_DIR / f"raw_slice_{idx}.mp4"
        safe_title = sanitize_filename(clip["title"])
        category = clip.get("category", "default").lower()
        preset = CATEGORY_TO_PRESET.get(category, "default")
        final_file = OUTPUT_DIR / f"clip_{idx:02d}_{preset}_{safe_title}.mp4"

        start_hms = clip.get("start_hms") or seconds_to_hms(clip["start_sec"])
        end_hms = clip.get("end_hms") or seconds_to_hms(clip["end_sec"])

        if not download_clip_slice(url, start_hms, end_hms, raw_file):
            print(f"[Clipper] Clip #{idx} download failed — skipping.")
            generated.append({"index": idx, "success": False, "clip": clip})
            continue

        success = render_vertical_short(
            raw_path=raw_file,
            out_path=final_file,
            hook_text=clip.get("hook_overlay", clip["title"]),
            cues=cues,
            clip_start_sec=clip["start_sec"],
            clip_end_sec=clip["end_sec"],
            preset=preset,
        )

        if success:
            print(f"[Clipper] Saved -> {final_file}")
            generated.append({
                "index": idx,
                "success": True,
                "path": str(final_file),
                "clip": clip,
                "preset": preset,
            })
        else:
            print(f"[Clipper] Render failed for clip #{idx}")
            generated.append({"index": idx, "success": False, "clip": clip})

        raw_file.unlink(missing_ok=True)

    return generated