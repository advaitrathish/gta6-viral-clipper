"""
clipper.py — Video download, rendering, and editing for GTA6 AI Content Factory
Renders 1080x1920 (9:16) vertical Shorts with:
- YuNet face-centered webcam crop on top (1080x860) — fills the top pane cleanly without dark dead space
- Wide gameplay crop on bottom (1080x1060) — preserves ~20% more horizontal HUD/action
- Two-stage dynamic editorial callouts (hook_overlay -> mid_callout) + mid-clip reaction zoom punch-in
- Four editing presets: comedy, action, cinematic, discovery
- ASS caption burning & audio loudness normalization
"""
import re
import shutil
import statistics
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _write_text_file(text: str, tmp_dir: Path, filename: str = "hook_text.txt") -> Path:
    p = tmp_dir / filename
    p.write_text(text.strip().upper(), encoding="utf-8")
    return p


def get_yunet_detector(w: int, h: int):
    if not YUNET_PATH.exists():
        print("[Clipper] Downloading YuNet face detection model...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (w, h), 0.70, 0.3, 5000)


def detect_facecam_region(video_path: Path) -> Optional[Tuple[str, str]]:
    """
    Detects which screen corner holds the streamer webcam using YuNet and returns
    (corner_name, exact_ffmpeg_crop_expr) fitted to a 1080:860 (~1.256:1) top pane.
    """
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
    detections: Dict[str, List[Tuple[float, float, float, float]]] = {
        "top-right": [],
        "top-left": [],
        "bottom-right": [],
        "bottom-left": [],
    }

    for p in (0.15, 0.35, 0.55, 0.75, 0.90):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * p))
        ok, frame = cap.read()
        if not ok:
            continue

        _, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            # Pick the highest-confidence face
            face = max(faces, key=lambda f: float(f[-1]))
            fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
            cx = fx + (fw / 2.0)
            cy = fy + (fh / 2.0)
            is_right = cx >= (width * 0.5)
            is_bottom = cy >= (height * 0.5)

            if is_right and not is_bottom:
                detections["top-right"].append((cx, cy, fw, fh))
            elif not is_right and not is_bottom:
                detections["top-left"].append((cx, cy, fw, fh))
            elif is_right and is_bottom:
                detections["bottom-right"].append((cx, cy, fw, fh))
            else:
                detections["bottom-left"].append((cx, cy, fw, fh))

    cap.release()
    best_corner, boxes = max(detections.items(), key=lambda kv: len(kv[1]))
    if not boxes:
        return None

    med_cx = statistics.median(b[0] for b in boxes)
    med_cy = statistics.median(b[1] for b in boxes)
    med_fw = statistics.median(b[2] for b in boxes)
    med_fh = statistics.median(b[3] for b in boxes)

    # Frame shoulders + webcam background around detected face at 1080:860 (1.2558:1) aspect ratio
    crop_h = max(int(height * 0.24), min(int(height * 0.38), int(med_fh * 3.1)))
    crop_w = min(int(width * 0.44), int(crop_h * (1080.0 / 860.0)))
    crop_h = int(crop_w * (860.0 / 1080.0))

    # Shift center slightly downward (+0.15*fh) so head + shoulders sit comfortably inside frame
    x1 = int(med_cx - (crop_w / 2.0))
    y1 = int((med_cy + med_fh * 0.15) - (crop_h / 2.0))

    # Clamp strictly to the detected corner quadrant so gameplay doesn't bleed in
    if "right" in best_corner:
        x1 = max(int(width * 0.55), min(width - crop_w, x1))
    else:
        x1 = max(0, min(int(width * 0.45) - crop_w, x1))

    if "bottom" in best_corner:
        y1 = max(int(height * 0.55), min(height - crop_h, y1))
    else:
        y1 = max(0, min(int(height * 0.45) - crop_h, y1))

    crop_expr = f"crop={crop_w}:{crop_h}:{x1}:{y1}"
    return best_corner, crop_expr


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
    mid_text: str = "",
) -> bool:
    ff_exe = get_ffmpeg_exe()
    style = PRESET_STYLES.get(preset, PRESET_STYLES["default"])
    eq_filt = _build_eq_filter(style["saturation"], style["contrast"])
    use_nvenc = _has_nvenc(ff_exe)
    trim_end = _detect_trim_end(ff_exe, raw_path)

    effective_dur = trim_end if trim_end else max(20.0, clip_end_sec - clip_start_sec)
    mid_sec = round(effective_dur * 0.48, 2)

    facecam_info = detect_facecam_region(raw_path)

    with tempfile.TemporaryDirectory(prefix="gta6_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        ass_path = tmp_path / "captions.ass"
        font_name = Path(FONT_PATH).stem if Path(FONT_PATH).exists() else "Liberation Sans"
        has_caps = generate_ass_captions(
            cues, ass_path, clip_start_sec, clip_end_sec, font_name, 78, preset
        )

        hook_file = _write_text_file(hook_text, tmp_path, "hook_text.txt")
        hook_y = "822" if facecam_info else "240"

        hook_file_escaped = _escape_ffmpeg_filter_path(hook_file)
        font_part = ""
        if Path(FONT_PATH).exists():
            font_escaped = _escape_ffmpeg_filter_path(Path(FONT_PATH))
            font_part = f"fontfile='{font_escaped}':"

        enable_first = f":enable='lt(t,{mid_sec})'" if mid_text.strip() else ""
        drawtext_1 = (
            f"drawtext=textfile='{hook_file_escaped}':"
            f"{font_part}"
            f"fontcolor=white:fontsize=46:"
            f"box=1:boxcolor=black@0.85:boxborderw=14:"
            f"x=(w-text_w)/2:y={hook_y}:"
            f"fix_bounds=1{enable_first}"
        )

        drawtext_2 = ""
        if mid_text.strip():
            mid_file = _write_text_file(mid_text, tmp_path, "mid_text.txt")
            mid_file_escaped = _escape_ffmpeg_filter_path(mid_file)
            drawtext_2 = (
                f",drawtext=textfile='{mid_file_escaped}':"
                f"{font_part}"
                f"fontcolor=#FFD700:fontsize=46:"
                f"box=1:boxcolor=black@0.85:boxborderw=14:"
                f"x=(w-text_w)/2:y={hook_y}:"
                f"fix_bounds=1:enable='gte(t,{mid_sec})'"
            )

        ass_escaped = _escape_ffmpeg_filter_path(ass_path)
        ass_filter = f"ass='{ass_escaped}'" if has_caps else ""

        if facecam_info:
            corner, cam_crop = facecam_info
            layout_label = f"Face-Centered Webcam ({corner}: {cam_crop}) + Wide Gameplay"
        else:
            cam_crop = ""
            layout_label = "Full-Frame Gameplay (No Facecam)"
        print(f"[Clipper] Layout -> {layout_label} | Preset -> {preset.upper()}")

        # Opening hook zoom + subtle mid-clip reaction punch-in for editorial dynamism
        dynamic_zoom = (
            f"scale=w='if(lt(t,2.5),iw*(1+0.07*(1-t/2.5)),if(between(t,{mid_sec},{mid_sec + 3.0}),iw*1.05,iw))':"
            f"h='if(lt(t,2.5),ih*(1+0.07*(1-t/2.5)),if(between(t,{mid_sec},{mid_sec + 3.0}),ih*1.05,ih))':eval=frame,"
            f"crop=1080:1920:(iw-1080)/2:(ih-1920)/2"
        )

        if facecam_info:
            # Top: 1080x860 face-centered webcam fill (no dead space on left/right)
            # Bottom: 1080x1060 wider gameplay crop (preserves ~20% more horizontal game view)
            filter_parts = [
                f"[0:v]{cam_crop},{eq_filt},scale=1080:860:force_original_aspect_ratio=increase,"
                f"crop=1080:860,setsar=1[cam]",
                f"[0:v]crop=ih*27/20:ih:(iw-ih*27/20)/2:0,{eq_filt},scale=1080:1060:force_original_aspect_ratio=increase,"
                f"crop=1080:1060,setsar=1[game]",
                f"[cam][game]vstack=inputs=2,{dynamic_zoom}[stacked]",
            ]
            last_label = "stacked"
        else:
            filter_parts = [
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,boxblur=25:10,{eq_filt},eq=brightness=-0.12[bg]",
                f"[0:v]{eq_filt},scale=1080:-2[fg]",
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{dynamic_zoom}[comp]",
            ]
            last_label = "comp"

        if ass_filter:
            filter_parts.append(f"[{last_label}]{ass_filter}[captioned]")
            last_label = "captioned"

        filter_parts.append(f"[{last_label}]{drawtext_1}{drawtext_2}[outv]")
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
            mid_text=clip.get("mid_callout", ""),
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