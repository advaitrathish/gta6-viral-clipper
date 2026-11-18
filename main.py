"""
main.py — Single-command entry point for GTA6 AI Content Factory

Usage:
  python main.py "YOUTUBE_URL" [options]

Options:
  --clips N      Number of clips to ask Gemini to find (default: 5)
  --top N        Top N clips to actually download and render (default: 3)
  --out DIR      Output directory (default: output_clips/)
  --force        Clear cache and re-process from scratch
  --no-cache     Same as --force (alias)
  --debug        Enable verbose debug logging

Example:
  python main.py "https://www.youtube.com/watch?v=XXXX" --clips 7 --top 3 --out my_clips
"""
import argparse
import sys
from pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="GTA6 AI Content Factory — Automated YouTube Shorts Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        help="YouTube URL of the gaming VOD or stream to process",
    )
    parser.add_argument(
        "--clips", type=int, default=5, metavar="N",
        help="Number of viral moment candidates to ask Gemini to identify (default: 5)",
    )
    parser.add_argument(
        "--top", type=int, default=3, metavar="N",
        help="Number of top clips to download and render (default: 3)",
    )
    parser.add_argument(
        "--out", type=str, default=None, metavar="DIR",
        help="Output directory for rendered clips (default: output_clips/)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cache and re-process from scratch",
    )
    parser.add_argument(
        "--no-cache", dest="no_cache", action="store_true",
        help="Alias for --force",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    print("=" * 56)
    print("  GTA6 AI Content Factory  --  Shorts MVP")
    print("=" * 56)

    result = run_pipeline(
        url       = args.url,
        max_clips = args.clips,
        top_n     = args.top,
        out_dir   = args.out,
        force     = args.force or args.no_cache,
        debug     = args.debug,
    )

    # Print final summary
    print(result.get("summary", ""))

    # Exit code: 0 = success, 1 = no clips produced
    if result.get("success") and result.get("clips_built", 0) > 0:
        print(f"\n✅ Done! {result['clips_built']} clip(s) ready for review.")
        return 0
    else:
        print("\n❌ Pipeline completed but no valid clips were produced.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
