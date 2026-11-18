"""
drive_uploader.py — Secure Google Drive Uploader for GTA6 Viral Clipper
Uploads rendered Shorts and metadata to the 'GTA6_Viral_Shorts' folder.
"""
import os
import base64
import json
import logging
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

logger = logging.getLogger("drive_uploader")

# Target Google Drive Folder ID for GTA6_Viral_Shorts
FOLDER_ID = "1k4FgP4i3acjA7IG_cWtk4SD6W0ZD4Rm5"


def get_drive_service():
    """Build and return authenticated Google Drive API service using base64 token secret."""
    token_b64 = os.getenv("GDRIVE_TOKEN_B64")
    if not token_b64:
        raise RuntimeError("GDRIVE_TOKEN_B64 environment variable is missing or empty.")

    try:
        token_json = base64.b64decode(token_b64).decode("utf-8")
        token_data = json.loads(token_json)
        creds = Credentials.from_authorized_user_info(token_data)
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Google Drive credentials from GDRIVE_TOKEN_B64: {e}")


def upload_clip_to_drive(file_path: Path) -> str:
    """
    Uploads an .mp4 file or its corresponding .json metadata to the 'GTA6_Viral_Shorts' Drive folder.
    Returns the webViewLink URL of the uploaded file.
    """
    if not file_path.exists():
        logger.warning(f"File not found for Drive upload: {file_path}")
        return ""

    service = get_drive_service()
    file_name = file_path.name
    
    # Escape single quotes safely for Google Drive query syntax
    safe_file_name = file_name.replace("'", "\\'")
    mime_type = "video/mp4" if file_path.suffix.lower() == ".mp4" else "application/json"

    try:
        # Check if file already exists in the folder
        query = f"name = '{safe_file_name}' and '{FOLDER_ID}' in parents and trashed = false"
        results = service.files().list(q=query, spaces="drive", fields="files(id, webViewLink)").execute()
        files = results.get("files", [])

        media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)

        if files:
            file_id = files[0]["id"]
            logger.info(f"Updating existing file on Google Drive: {file_name}")
            updated_file = service.files().update(
                fileId=file_id,
                media_body=media,
                fields="id, webViewLink"
            ).execute()
            link = updated_file.get("webViewLink", "")
            logger.info(f"Successfully updated on Drive: {link}")
            return link
        else:
            logger.info(f"Uploading new file to Google Drive: {file_name}")
            file_metadata = {
                "name": file_name,
                "parents": [FOLDER_ID]
            }
            new_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, webViewLink"
            ).execute()
            link = new_file.get("webViewLink", "")
            logger.info(f"Successfully uploaded to Drive: {link}")
            return link

    except Exception as e:
        logger.error(f"Google Drive upload failed for {file_name}: {e}")
        return ""