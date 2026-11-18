"""
telegram_trigger.py — Mobile phone control + auto-delivery of rendered Shorts
Send a YouTube link from your phone -> PC extracts, analyzes, renders -> sends MP4s back to Telegram!
"""
import asyncio
import logging
import sys
import threading
from pathlib import Path

try:
    from telegram import Update
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        filters, ContextTypes
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USER, OUTPUT_DIR

logger = logging.getLogger("telegram_trigger")

_job_state = {
    "running": False,
    "url": None,
    "status": "idle",
    "clips": [],
}
_pending_url_from_user: dict = {}


def _is_authorized(update: Update) -> bool:
    allowed = TELEGRAM_ALLOWED_USER.strip()
    if not allowed:
        logger.warning("TELEGRAM_ALLOWED_USER not set — denying all requests.")
        return False
    return str(update.effective_user.id) == allowed


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    await update.message.reply_text(
        "*GTA6 AI Content Factory — Mobile Control*\n\n"
        "Paste any YouTube URL directly (or use `/run`) to generate Shorts.\n\n"
        "Commands:\n"
        "/run — Start a new clip job\n"
        "/status — Check current job\n"
        "/clips — List & resend generated clips\n"
        "/approve N — Approve clip #N for upload\n"
        "/reject N — Reject clip #N",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    if _job_state["running"]:
        await update.message.reply_text(
            f"A job is already running for:\n{_job_state['url']}\n\nStatus: {_job_state['status']}"
        )
        return
    user_id = str(update.effective_user.id)
    _pending_url_from_user[user_id] = True
    await update.message.reply_text(
        "Send me the YouTube URL to process (e.g. https://www.youtube.com/watch?v=XXXX)"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    if _job_state["running"]:
        await update.message.reply_text(
            f"Processing: {_job_state['url']}\nStatus: {_job_state['status']}"
        )
    else:
        await update.message.reply_text(
            f"Idle. Last status: {_job_state['status']}"
        )


async def cmd_clips(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    clips = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not clips:
        await update.message.reply_text("No clips in output_clips/ yet.")
        return
    lines = [f"Generated clips ({len(clips)} found):"]
    for i, c in enumerate(clips[:10], 1):
        size_mb = c.stat().st_size / 1_048_576
        lines.append(f"  #{i}: {c.name} ({size_mb:.1f} MB)")
    await update.message.reply_text("\n".join(lines))


async def cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    clips = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    args = context.args
    if args and args[0].isdigit():
        idx = int(args[0]) - 1
        if 0 <= idx < len(clips):
            clip = clips[idx]
            (clip.parent / (clip.stem + ".approved")).touch()
            await update.message.reply_text(f"Approved: {clip.name}")
            return
    await update.message.reply_text("Usage: /approve N (use /clips to see numbers)")


async def cmd_reject(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    clips = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    args = context.args
    if args and args[0].isdigit():
        idx = int(args[0]) - 1
        if 0 <= idx < len(clips):
            clip = clips[idx]
            (clip.parent / (clip.stem + ".rejected")).touch()
            await update.message.reply_text(f"Rejected: {clip.name}")
            return
    await update.message.reply_text("Usage: /reject N (use /clips to see numbers)")


async def _deliver_results_to_chat(bot, chat_id: int, result: dict):
    """Send pipeline summary and rendered MP4 files directly to Telegram chat."""
    passed = [
        r for r in result.get("results", [])
        if r.get("success") and r.get("qc", {}).get("ok") and r.get("path")
    ]
    if not passed:
        await bot.send_message(
            chat_id=chat_id,
            text=f"Pipeline finished with 0 valid clips.\nSummary:\n{result.get('summary', 'Unknown error')}",
        )
        return

    await bot.send_message(
        chat_id=chat_id,
        text=f"Done in {result.get('elapsed_sec', '?')}s! Uploading {len(passed)} clip(s) to your phone...",
    )

    for r in passed:
        video_path = Path(r["path"])
        clip = r.get("clip", {})
        if not video_path.exists():
            continue

        size_mb = video_path.stat().st_size / 1_048_576
        hashtags = " ".join(clip.get("hashtags", ["#GTA6", "#Shorts"]))
        caption = (
            f"#{r['index']} | Score: {clip.get('virality_score', '?')}/100 ({clip.get('category', '').upper()})\n"
            f"Title: {clip.get('title', video_path.stem)}\n"
            f"Hook: {clip.get('hook_overlay', '')}\n\n"
            f"{hashtags}"
        )[:1000]

        if size_mb > 49.0:
            await bot.send_message(
                chat_id=chat_id,
                text=f"Clip #{r['index']} is {size_mb:.1f} MB (exceeds 50 MB Telegram limit). Saved locally at:\n{video_path}",
            )
            continue

        try:
            with video_path.open("rb") as vf:
                await bot.send_video(
                    chat_id=chat_id,
                    video=vf,
                    caption=caption,
                    supports_streaming=True,
                    read_timeout=120,
                    write_timeout=120,
                )
        except Exception as e:
            logger.error(f"Failed to send video {video_path.name}: {e}")
            await bot.send_message(
                chat_id=chat_id,
                text=f"Rendered {video_path.name} ({size_mb:.1f} MB), but Telegram upload failed: {e}",
            )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return

    user_id = str(update.effective_user.id)
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()

    is_yt_link = text.startswith(("https://www.youtube.com/", "https://youtu.be/", "https://m.youtube.com/"))
    if _pending_url_from_user.pop(user_id, False) or is_yt_link:
        if not text.startswith("http"):
            await update.message.reply_text("That doesn't look like a URL. Use /run to try again.")
            return
        if _job_state["running"]:
            await update.message.reply_text(
                f"Already processing:\n{_job_state['url']}\nStatus: {_job_state['status']}"
            )
            return

        url = text
        loop = asyncio.get_running_loop()
        bot = context.bot

        await update.message.reply_text(
            f"Starting pipeline for:\n{url}\n\nExtracting, analyzing with Gemini 4 Hydrogen, and rendering top 3 Shorts..."
        )

        def _worker():
            _job_state["running"] = True
            _job_state["url"] = url
            _job_state["status"] = "running pipeline..."
            try:
                from pipeline import run_pipeline
                result = run_pipeline(url=url, max_clips=5, top_n=3)
                _job_state["status"] = "done"
                _job_state["clips"] = [
                    r.get("path") for r in result.get("results", [])
                    if r.get("success")
                ]
                asyncio.run_coroutine_threadsafe(
                    _deliver_results_to_chat(bot, chat_id, result),
                    loop,
                )
            except Exception as e:
                _job_state["status"] = f"error: {e}"
                logger.error(f"Pipeline error: {e}")
                asyncio.run_coroutine_threadsafe(
                    bot.send_message(chat_id=chat_id, text=f"Pipeline error: {e}"),
                    loop,
                )
            finally:
                _job_state["running"] = False

        threading.Thread(target=_worker, daemon=True).start()
    else:
        await update.message.reply_text(
            "Paste a YouTube URL directly or use /run to start a new job."
        )


def run_bot():
    if not TELEGRAM_AVAILABLE:
        print("python-telegram-bot not installed. Run: pip install python-telegram-bot")
        sys.exit(1)

    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    if not TELEGRAM_ALLOWED_USER:
        print("TELEGRAM_ALLOWED_USER not set — bot will refuse all requests!")

    logging.basicConfig(level=logging.INFO)
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("run", cmd_run))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clips", cmd_clips))
    app.add_handler(CommandHandler("approve", cmd_approve))
    app.add_handler(CommandHandler("reject", cmd_reject))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print(f"Telegram bot started. Authorized user: {TELEGRAM_ALLOWED_USER or '(none)'}")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run_bot()