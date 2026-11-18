"""
pipeline.py — Orchestration layer for GTA6 AI Content Factory
Coordinates: extract → cache → analyze → validate → download → render → QC → metadata
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    OUTPUT_DIR, LOGS_DIR, MIN_CLIP_DURATION, MAX_CLIP_DURATION, DEBUG
)

logger = logging.getLogger("gta6_pipeline")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(debug: bool = False) -> None:
    level    = logging.DEBUG if debug else logging.INFO
    log_file = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger.info(f"Log file: {log_file}")


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def _save_metadata(clip: Dict, output_path: str, source_url: str, source_title: str, preset: str) -> Path:
    """Write a metadata.json beside the rendered clip."""
    meta = {
        "source_url":        source_url,
        "source_title":      source_title,
        "source_video_id":   clip.get("video_id", ""),
        "clip_start_sec":    clip.get("start_sec"),
        "clip_end_sec":      clip.get("end_sec"),
        "duration_sec":      clip.get("duration"),
        "title":             clip.get("title"),
        "hook_overlay":      clip.get("hook_overlay"),
        "alternative_hooks": clip.get("alternative_hooks", []),
        "description":       clip.get("description", ""),
        "hashtags":          clip.get("hashtags", []),
        "virality_score":    clip.get("virality_score"),
        "confidence_score":  clip.get("confidence_score"),
        "category":          clip.get("category"),
        "editing_preset":    preset,
        "output_path":       output_path,
        "generated_at":      datetime.utcnow().isoformat() + "Z",
        "rights_note":       (
            "Uses authorized gameplay footage only. "
            "Manual review required before publishing."
        ),
    }
    out = Path(output_path)
    meta_path = out.parent / (out.stem + ".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    url:       str,
    max_clips: int = 5,
    top_n:     int = 3,
    out_dir:   Optional[str] = None,
    force:     bool = False,
    debug:     bool = False,
) -> Dict:
    """
    Full end-to-end pipeline.

    Returns:
      {
        "success":    bool,
        "clips_built": int,
        "results":    [result_dict, ...],
        "summary":    str,
      }
    """
    setup_logging(debug)
    t0 = time.time()

    # Redirect output directory if specified
    if out_dir:
        from config import OUTPUT_DIR as _OD
        import config as _cfg
        _cfg.OUTPUT_DIR = Path(out_dir)
        _cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Also update clipper's OUTPUT_DIR reference
        try:
            import clipper as _clipper
            _clipper.OUTPUT_DIR = _cfg.OUTPUT_DIR
        except Exception:
            pass

    effective_out = Path(out_dir) if out_dir else OUTPUT_DIR

    # ------------------------------------------------------------------ #
    # Step 1: Validate URL                                                 #
    # ------------------------------------------------------------------ #
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return {"success": False, "clips_built": 0, "results": [], "summary": "Invalid URL"}

    logger.info(f"=== GTA6 AI Content Factory — Pipeline Start ===")
    logger.info(f"URL:        {url}")
    logger.info(f"Max clips:  {max_clips} | Top N: {top_n} | Force: {force}")

    # ------------------------------------------------------------------ #
    # Step 2: Extract signals (or load from cache)                        #
    # ------------------------------------------------------------------ #
    from cache_manager import load_signals, save_signals, load_clips, save_clips, clear_cache
    from extractor import fetch_video_signals

    if force:
        clear_cache(url)

    signals = None
    if not force:
        signals = load_signals(url)

    if signals is None:
        logger.info("--- Step 2: Extracting signals from YouTube ---")
        signals = fetch_video_signals(url)
        save_signals(url, signals)
    else:
        logger.info("--- Step 2: Signals loaded from cache ---")

    source_title = signals.get("title", "GTA VI Gaming VOD")
    duration_sec = signals.get("duration_sec", 0.0)
    logger.info(f"Title:      {source_title}")
    logger.info(f"Duration:   {duration_sec:.0f}s | "
                f"Heatmap peaks: {len(signals.get('heatmap_peaks', []))} | "
                f"Cues: {len(signals.get('cues', []))}")

    # ------------------------------------------------------------------ #
    # Step 3: Gemini analysis (or load from cache)                        #
    # ------------------------------------------------------------------ #
    clips = None
    if not force:
        clips = load_clips(url)

    if clips is None:
        logger.info(f"--- Step 3: Analyzing with Gemini (asking for {max_clips} clips) ---")
        from analyzer import find_viral_clips
        try:
            clips = find_viral_clips(signals, max_clips=max_clips)
            save_clips(url, clips)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {"success": False, "clips_built": 0, "results": [], "summary": str(e)}
    else:
        logger.info(f"--- Step 3: {len(clips)} clips loaded from Gemini cache ---")

    if not clips:
        msg = "No viral clips identified by Gemini."
        logger.warning(msg)
        return {"success": False, "clips_built": 0, "results": [], "summary": msg}

    # Print analysis summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Gemini identified {len(clips)} candidate clips:")
    for i, c in enumerate(clips, 1):
        logger.info(
            f"  #{i} [{c.get('start_hms','?')} → {c.get('end_hms','?')} | "
            f"{c.get('duration','?')}s | "
            f"Score: {c.get('virality_score','?')}/100 | "
            f"{c.get('category','?').upper()}] "
            f"{c.get('title','?')}"
        )
    logger.info(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Step 4: Download + Render top_n clips                               #
    # ------------------------------------------------------------------ #
    logger.info(f"--- Step 4: Building top {top_n} clips ---")
    from clipper import build_clips

    cues   = signals.get("cues", [])
    built  = build_clips(url, clips, cues, top_n=top_n)

    # ------------------------------------------------------------------ #
    # Step 5: Quality check                                                #
    # ------------------------------------------------------------------ #
    logger.info("--- Step 5: Running quality checks ---")
    from quality_checker import check_video

    final_results = []
    for item in built:
        if not item.get("success") or not item.get("path"):
            final_results.append({**item, "qc": {"ok": False, "issues": ["Render failed"]}})
            continue

        path   = item["path"]
        qc     = check_video(Path(path), MIN_CLIP_DURATION - 5, MAX_CLIP_DURATION + 7)
        preset = item.get("preset", "default")
        clip   = item["clip"]

        if qc["ok"]:
            # Save metadata JSON
            meta_path = _save_metadata(clip, path, url, source_title, preset)
            logger.info(f"✓ Clip #{item['index']} passed QC → metadata: {meta_path.name}")
        else:
            logger.warning(f"✗ Clip #{item['index']} FAILED QC: {qc['issues']}")

        final_results.append({**item, "qc": qc})

    # ------------------------------------------------------------------ #
    # Step 6: Final summary                                                #
    # ------------------------------------------------------------------ #
    passed = [r for r in final_results if r.get("qc", {}).get("ok")]
    failed = [r for r in final_results if not r.get("qc", {}).get("ok")]
    elapsed = round(time.time() - t0, 1)

    summary_lines = [
        "",
        "=" * 65,
        f"  GTA6 AI Content Factory — Pipeline Complete ({elapsed}s)",
        "=" * 65,
        f"  Source:  {source_title}",
        f"  URL:     {url[:60]}{'...' if len(url) > 60 else ''}",
        f"  Result:  {len(passed)} clip(s) ready / {len(failed)} failed",
        "-" * 65,
    ]
    for r in passed:
        c   = r["clip"]
        qc  = r.get("qc", {})
        summary_lines.append(
            f"  ✓ #{r['index']:02d} | {c.get('category','?').upper():12} | "
            f"{qc.get('duration', c.get('duration', '?'))}s | "
            f"Score {c.get('virality_score','?'):3}/100 | "
            f"{c.get('title','?')[:40]}"
        )
        summary_lines.append(f"       → {r.get('path','?')}")
    for r in failed:
        c = r.get("clip", {})
        issues = r.get("qc", {}).get("issues", [r.get("reason", "unknown error")])
        summary_lines.append(f"  ✗ #{r['index']:02d} | FAILED | {c.get('title','?')[:40]}")
        for iss in issues[:2]:
            summary_lines.append(f"       → {iss}")
    summary_lines.append("=" * 65)

    summary = "\n".join(summary_lines)
    logger.info(summary)

    return {
        "success":     len(passed) > 0,
        "clips_built": len(passed),
        "results":     final_results,
        "summary":     summary,
        "elapsed_sec": elapsed,
    }
