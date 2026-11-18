"""
publisher.py — Optional YouTube Shorts uploader for GTA6 AI Content Factory

Requirements:
  pip install google-api-python-client google-auth-oauthlib google-auth-httplib2

Setup:
  1. Go to Google Cloud Console → enable YouTube Data API v3
  2. Create OAuth 2.0 credentials (Desktop app)
  3. Download credentials.json to project root
  4. First run will open browser for auth → saves token.json

Usage:
  python publisher.py output_clips/clip_01_comedy_Title.mp4 --privacy unlisted
"""
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

from config import YT_CLIENT_SECRETS, YT_TOKEN_FILE

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_credentials():
    """Get or refresh YouTube OAuth2 credentials."""
    creds = None

    if YT_TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(YT_TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not YT_CLIENT_SECRETS.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {YT_CLIENT_SECRETS}. "
                    "Download it from Google Cloud Console."
                )
            flow  = InstalledAppFlow.from_client_secrets_file(str(YT_CLIENT_SECRETS), SCOPES)
            creds = flow.run_local_server(port=0)
        YT_TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return creds


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_short(
    video_path: Path,
    title: str,
    description: str,
    tags: list,
    privacy: str = "unlisted",
) -> Optional[str]:
    """
    Upload a video to YouTube as a Short.

    Args:
        video_path:  Path to the MP4 file
        title:       Video title (max 100 chars)
        description: Video description
        tags:        List of tag strings (hashtags without #)
        privacy:     "public", "unlisted", or "private"

    Returns:
        YouTube video ID on success, None on failure
    """
    if not YOUTUBE_AVAILABLE:
        print("❌ YouTube upload libraries not installed.")
        print("   Run: pip install google-api-python-client google-auth-oauthlib")
        return None

    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return None

    # Require manual approval before uploading
    approval_marker = video_path.parent / (video_path.stem + ".approved")
    if not approval_marker.exists():
        print(
            f"⚠️  Clip not approved for upload: {video_path.name}\n"
            f"   Create {approval_marker.name} to approve, or use /approve N in Telegram."
        )
        return None

    print(f"[Publisher] Authenticating with YouTube...")
    try:
        creds   = _get_credentials()
        youtube = build("youtube", "v3", credentials=creds)
    except Exception as e:
        print(f"❌ YouTube authentication failed: {e}")
        return None

    # Clean tags (remove # prefix, limit)
    clean_tags = [t.lstrip("#") for t in tags][:500]

    body = {
        "snippet": {
            "title":       title[:100],
            "description": description[:5000],
            "tags":        clean_tags,
            "categoryId":  "20",   # Gaming
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype   = "video/mp4",
        resumable  = True,
        chunksize  = 1024 * 1024 * 5,  # 5 MB chunks
    )

    print(f"[Publisher] Uploading: {video_path.name}")
    print(f"            Title:    {title}")
    print(f"            Privacy:  {privacy}")

    try:
        request  = youtube.videos().insert(
            part = "snippet,status",
            body = body,
            media_body = media,
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"            Progress: {int(status.progress() * 100)}%")

        video_id = response.get("id")
        print(f"[Publisher] ✓ Uploaded! https://youtube.com/shorts/{video_id}")

        # Save returned video ID next to clip
        result_path = video_path.parent / (video_path.stem + ".upload.json")
        result_path.write_text(
            json.dumps({"video_id": video_id, "url": f"https://youtube.com/shorts/{video_id}"},
                       indent=2),
            encoding="utf-8"
        )
        return video_id

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


def upload_approved_clips(out_dir: Path, privacy: str = "unlisted") -> None:
    """Find all approved clips in out_dir and upload them."""
    approved = [
        f for f in out_dir.glob("*.mp4")
        if (out_dir / (f.stem + ".approved")).exists()
        and not (out_dir / (f.stem + ".upload.json")).exists()
    ]

    if not approved:
        print("No approved clips awaiting upload.")
        return

    for video_path in approved:
        # Try to load metadata
        meta_path = video_path.parent / (video_path.stem + ".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = {
                "title":       video_path.stem,
                "description": "#GTA6 #Shorts",
                "hashtags":    ["#GTA6", "#Shorts"],
            }

        upload_short(
            video_path  = video_path,
            title       = meta.get("title", video_path.stem),
            description = meta.get("description", "") + "\n\n" + " ".join(meta.get("hashtags", [])),
            tags        = meta.get("hashtags", []),
            privacy     = privacy,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload approved clips to YouTube Shorts")
    parser.add_argument("path", nargs="?", help="Single clip to upload, or leave empty to upload all approved clips")
    parser.add_argument("--privacy", default="unlisted", choices=["public", "unlisted", "private"])
    parser.add_argument("--dir", default="output_clips", help="Directory to scan for approved clips")
    args = parser.parse_args()

    if args.path:
        p = Path(args.path)
        meta_p = p.parent / (p.stem + ".json")
        meta   = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        upload_short(p, meta.get("title", p.stem), meta.get("description", ""), meta.get("hashtags", []), args.privacy)
    else:
        upload_approved_clips(Path(args.dir), args.privacy)
