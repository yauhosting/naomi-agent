"""
NAOMI Agent - Google Calendar Client
Uses Google Calendar API via google-api-python-client.
Shares OAuth2 credentials with Gmail client.

Setup: Same as Gmail — credentials.json in data/ directory.
Calendar API must be enabled in Google Cloud Console.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger("naomi.calendar")

DATA_DIR = Path(__file__).parent.parent / "data"
CREDENTIALS_PATH = DATA_DIR / "gmail_credentials.json"
TOKEN_PATH = DATA_DIR / "calendar_token.json"

# Fallback: OpenClaw pre-migration credentials
_OPENCLAW_CREDS = Path.home() / ".openclaw.pre-migration" / "credentials" / "google_credentials.json"
_OPENCLAW_CREDS_V2 = Path.home() / ".openclaw" / "credentials" / "google_credentials.json"

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


class CalendarClient:
    """Google Calendar API client."""

    def __init__(self):
        self._service = None

    @property
    def available(self) -> bool:
        return (CREDENTIALS_PATH.exists()
                or _OPENCLAW_CREDS.exists()
                or _OPENCLAW_CREDS_V2.exists())

    def _ensure_deps(self):
        try:
            import google.auth  # noqa: F401
        except ImportError:
            import subprocess
            import sys
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install",
                 "google-api-python-client", "google-auth-httplib2",
                 "google-auth-oauthlib"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _get_service(self):
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

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                cred_path = None
                for p in [CREDENTIALS_PATH, _OPENCLAW_CREDS, _OPENCLAW_CREDS_V2]:
                    if p.exists():
                        cred_path = p
                        break
                if not cred_path:
                    raise FileNotFoundError(
                        f"Calendar credentials not found. Checked:\n"
                        f"  {CREDENTIALS_PATH}\n  {_OPENCLAW_CREDS}\n  {_OPENCLAW_CREDS_V2}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(cred_path), SCOPES
                )
                creds = flow.run_local_server(port=8080)

            TOKEN_PATH.write_text(creds.to_json())

        self._service = build("calendar", "v3", credentials=creds)
        return self._service

    def list_events(
        self,
        days: int = 7,
        max_results: int = 20,
        calendar_id: str = "primary",
    ) -> List[Dict[str, Any]]:
        """List upcoming events within N days."""
        try:
            service = self._get_service()
            now = datetime.now(timezone.utc)
            time_max = now + timedelta(days=days)

            result = service.events().list(
                calendarId=calendar_id,
                timeMin=now.isoformat(),
                timeMax=time_max.isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            events = []
            for ev in result.get("items", []):
                start = ev.get("start", {})
                end = ev.get("end", {})
                events.append({
                    "id": ev["id"],
                    "summary": ev.get("summary", "(no title)"),
                    "start": start.get("dateTime", start.get("date", "")),
                    "end": end.get("dateTime", end.get("date", "")),
                    "location": ev.get("location", ""),
                    "description": ev.get("description", "")[:200],
                    "status": ev.get("status", ""),
                    "html_link": ev.get("htmlLink", ""),
                })
            return events
        except Exception as e:
            logger.error("Calendar list error: %s", e)
            return []

    def today_events(self, calendar_id: str = "primary") -> List[Dict[str, Any]]:
        """Get today's events."""
        return self.list_events(days=1, calendar_id=calendar_id)

    def create_event(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        description: str = "",
        location: str = "",
        calendar_id: str = "primary",
    ) -> Optional[Dict[str, Any]]:
        """Create a calendar event.

        Args:
            start_time: ISO format, e.g. "2026-04-15T10:00:00+08:00"
            end_time: ISO format
        """
        try:
            service = self._get_service()
            event_body: Dict[str, Any] = {
                "summary": summary,
                "start": {"dateTime": start_time},
                "end": {"dateTime": end_time},
            }
            if description:
                event_body["description"] = description
            if location:
                event_body["location"] = location

            created = service.events().insert(
                calendarId=calendar_id, body=event_body
            ).execute()

            logger.info("Calendar event created: %s (%s)", summary, created["id"])
            return {
                "id": created["id"],
                "summary": created.get("summary"),
                "html_link": created.get("htmlLink"),
            }
        except Exception as e:
            logger.error("Calendar create error: %s", e)
            return None

    def delete_event(
        self, event_id: str, calendar_id: str = "primary"
    ) -> bool:
        """Delete a calendar event."""
        try:
            service = self._get_service()
            service.events().delete(
                calendarId=calendar_id, eventId=event_id
            ).execute()
            logger.info("Calendar event deleted: %s", event_id)
            return True
        except Exception as e:
            logger.error("Calendar delete error: %s", e)
            return False

    def quick_add(
        self, text: str, calendar_id: str = "primary"
    ) -> Optional[Dict[str, Any]]:
        """Quick-add event using natural language.
        e.g. "Meeting with JW tomorrow at 3pm for 1 hour"
        """
        try:
            service = self._get_service()
            created = service.events().quickAdd(
                calendarId=calendar_id, text=text
            ).execute()
            return {
                "id": created["id"],
                "summary": created.get("summary"),
                "start": created.get("start", {}),
                "html_link": created.get("htmlLink"),
            }
        except Exception as e:
            logger.error("Calendar quickAdd error: %s", e)
            return None

    def search_events(
        self, query: str, days: int = 30, max_results: int = 10,
        calendar_id: str = "primary",
    ) -> List[Dict[str, Any]]:
        """Search events by keyword."""
        try:
            service = self._get_service()
            now = datetime.now(timezone.utc)
            time_max = now + timedelta(days=days)

            result = service.events().list(
                calendarId=calendar_id,
                timeMin=now.isoformat(),
                timeMax=time_max.isoformat(),
                q=query,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            return [
                {
                    "id": ev["id"],
                    "summary": ev.get("summary", ""),
                    "start": ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date", "")),
                }
                for ev in result.get("items", [])
            ]
        except Exception as e:
            logger.error("Calendar search error: %s", e)
            return []
