"""
NAOMI Agent - Gmail Client
Uses Gmail API via google-api-python-client.
Requires OAuth2 credentials (credentials.json) in data/ directory.

Setup:
1. Go to Google Cloud Console → APIs → Enable Gmail API
2. Create OAuth2 credentials (Desktop App)
3. Download credentials.json → data/gmail_credentials.json
4. First run will open browser for authorization
5. Token saved to data/gmail_token.json
"""
import os
import json
import base64
import logging
from typing import Optional, List, Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

logger = logging.getLogger("naomi.email")

DATA_DIR = Path(__file__).parent.parent / "data"
CREDENTIALS_PATH = DATA_DIR / "gmail_credentials.json"
TOKEN_PATH = DATA_DIR / "gmail_token.json"

# Fallback: OpenClaw pre-migration credentials
_OPENCLAW_CREDS = Path.home() / ".openclaw.pre-migration" / "credentials" / "google_credentials.json"
_OPENCLAW_CREDS_V2 = Path.home() / ".openclaw" / "credentials" / "google_credentials.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]


class GmailClient:
    """Gmail API client with OAuth2 authentication."""

    def __init__(self):
        self._service = None
        self._initialized = False

    @property
    def available(self) -> bool:
        """Check if Gmail credentials exist (check multiple paths)."""
        return (CREDENTIALS_PATH.exists()
                or _OPENCLAW_CREDS.exists()
                or _OPENCLAW_CREDS_V2.exists())

    def _ensure_deps(self):
        """Install Google API dependencies if missing."""
        try:
            import google.auth  # noqa: F401
        except ImportError:
            import subprocess
            import sys
            logger.info("Installing Google API dependencies...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install",
                 "google-api-python-client", "google-auth-httplib2",
                 "google-auth-oauthlib"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _get_service(self):
        """Get or create Gmail API service."""
        if self._service:
            return self._service

        self._ensure_deps()
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        creds = None
        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        # Check if existing token has all required scopes
        if creds and creds.valid:
            existing_scopes = set(creds.scopes or [])
            required_scopes = set(SCOPES)
            if not required_scopes.issubset(existing_scopes):
                logger.info("Gmail token missing scopes, re-authorizing: need %s",
                            required_scopes - existing_scopes)
                creds = None  # Force re-auth with full scopes

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Find credentials file from multiple paths
                cred_path = None
                for p in [CREDENTIALS_PATH, _OPENCLAW_CREDS, _OPENCLAW_CREDS_V2]:
                    if p.exists():
                        cred_path = p
                        break
                if not cred_path:
                    raise FileNotFoundError(
                        f"Gmail credentials not found. Checked:\n"
                        f"  {CREDENTIALS_PATH}\n  {_OPENCLAW_CREDS}\n  {_OPENCLAW_CREDS_V2}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(cred_path), SCOPES
                )
                creds = flow.run_local_server(port=18810)

            TOKEN_PATH.write_text(creds.to_json())

        self._service = build("gmail", "v1", credentials=creds)
        self._initialized = True
        return self._service

    def list_messages(
        self, query: str = "is:unread", max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """List messages matching a query."""
        try:
            service = self._get_service()
            result = service.users().messages().list(
                userId="me", q=query, maxResults=max_results
            ).execute()

            messages = result.get("messages", [])
            detailed = []
            for msg in messages:
                detail = service.users().messages().get(
                    userId="me", id=msg["id"], format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                ).execute()
                headers = {
                    h["name"]: h["value"]
                    for h in detail.get("payload", {}).get("headers", [])
                }
                detailed.append({
                    "id": msg["id"],
                    "thread_id": detail.get("threadId"),
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": detail.get("snippet", ""),
                    "labels": detail.get("labelIds", []),
                })
            return detailed
        except Exception as e:
            logger.error("Gmail list error: %s", e)
            return []

    def read_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Read full message content."""
        try:
            service = self._get_service()
            msg = service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()

            headers = {
                h["name"]: h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }

            body = self._extract_body(msg.get("payload", {}))

            return {
                "id": message_id,
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "subject": headers.get("Subject", ""),
                "date": headers.get("Date", ""),
                "body": body,
                "labels": msg.get("labelIds", []),
            }
        except Exception as e:
            logger.error("Gmail read error: %s", e)
            return None

    def send_message(
        self, to: str, subject: str, body: str, html: bool = False
    ) -> Optional[str]:
        """Send an email. Returns message ID on success."""
        try:
            service = self._get_service()
            if html:
                message = MIMEMultipart("alternative")
                message.attach(MIMEText(body, "html"))
            else:
                message = MIMEText(body)

            message["to"] = to
            message["subject"] = subject

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            sent = service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()

            logger.info("Email sent to %s: %s (id=%s)", to, subject, sent["id"])
            return sent["id"]
        except Exception as e:
            logger.error("Gmail send error: %s", e)
            return None

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search emails with Gmail query syntax."""
        return self.list_messages(query=query, max_results=max_results)

    def mark_read(self, message_id: str) -> bool:
        """Mark a message as read."""
        try:
            service = self._get_service()
            service.users().messages().modify(
                userId="me", id=message_id,
                body={"removeLabelIds": ["UNREAD"]},
            ).execute()
            return True
        except Exception as e:
            logger.error("Gmail mark_read error: %s", e)
            return False

    @staticmethod
    def _extract_body(payload: dict) -> str:
        """Extract text body from Gmail payload."""
        if payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

        parts = payload.get("parts", [])
        for part in parts:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            # Recurse into nested parts
            if part.get("parts"):
                result = GmailClient._extract_body(part)
                if result:
                    return result

        # Fallback: try HTML
        for part in parts:
            if part.get("mimeType") == "text/html":
                data = part.get("body", {}).get("data", "")
                if data:
                    html = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                    # Strip HTML tags (basic)
                    import re
                    return re.sub(r"<[^>]+>", "", html)[:3000]

        return ""
