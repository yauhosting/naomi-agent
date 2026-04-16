"""NAOMI Agent - Speech-to-Text Engine
Uses mlx-whisper for local Apple Silicon STT.
Falls back to OpenAI Whisper API.
"""
import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("naomi.stt")

# Lazy-loaded model
_whisper_model = None
_whisper_lock = threading.Lock()


async def transcribe(audio_path: str, language: str = "auto") -> Optional[str]:
    """Transcribe audio file to text.

    Tries in order:
    1. mlx-whisper (local, Apple Silicon optimized)
    2. OpenAI Whisper API (if OPENAI_API_KEY set)
    3. whisper CLI (if installed)

    Args:
        audio_path: Path to audio file (ogg, wav, mp3, etc.)
        language: Language hint ("zh", "en", "ja", "auto")
    Returns:
        Transcribed text, or None on failure.
    """
    if not os.path.exists(audio_path):
        logger.error("Audio file not found: %s", audio_path)
        return None

    # Ensure WAV format for best compatibility
    wav_path = audio_path
    if not audio_path.endswith(".wav"):
        wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", "16000", "-ac", "1", wav_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=30)
            if not os.path.exists(wav_path):
                wav_path = audio_path  # Fall back to original
        except Exception as e:
            logger.debug("ffmpeg conversion failed: %s", e)
            wav_path = audio_path

    # Try each method in order
    for method in [_mlx_whisper, _openai_whisper_api, _whisper_cli]:
        try:
            result = await method(wav_path, language)
            if result and result.strip():
                logger.info(
                    "Transcribed via %s: %s",
                    method.__name__,
                    result[:80],
                )
                return result.strip()
        except Exception as e:
            logger.debug("%s failed: %s", method.__name__, e)
            continue

    logger.warning("All STT methods failed for: %s", audio_path)
    return None


async def _mlx_whisper(audio_path: str, language: str) -> Optional[str]:
    """Use mlx-whisper for local transcription on Apple Silicon."""
    try:
        import mlx_whisper
    except ImportError:
        logger.debug("mlx-whisper not installed, skipping")
        return None

    lang_arg = None if language == "auto" else language
    model_repo = "mlx-community/whisper-large-v3-turbo"

    def _do_transcribe() -> str:
        global _whisper_model
        with _whisper_lock:
            kwargs = {"path_or_hf_repo": model_repo}
            if lang_arg:
                kwargs["language"] = lang_arg
            result = mlx_whisper.transcribe(audio_path, **kwargs)
            return result.get("text", "")

    text = await asyncio.to_thread(_do_transcribe)
    return text if text else None


async def _openai_whisper_api(audio_path: str, language: str) -> Optional[str]:
    """Use OpenAI Whisper API for transcription."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    try:
        import httpx
    except ImportError:
        return None

    lang_param = None if language == "auto" else language

    def _do_api_call() -> Optional[str]:
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # Determine filename extension for the API
        ext = Path(audio_path).suffix.lstrip(".")
        if ext not in ("wav", "mp3", "mp4", "m4a", "ogg", "webm", "flac"):
            ext = "wav"
        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "m4a": "audio/mp4",
            "mp4": "audio/mp4",
            "webm": "audio/webm",
        }
        mime = mime_map.get(ext, "audio/wav")

        files = {"file": (f"audio.{ext}", audio_data, mime)}
        data = {"model": "whisper-1"}
        if lang_param:
            data["language"] = lang_param

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
            )
            if resp.status_code == 200:
                return resp.json().get("text", "")
            else:
                logger.debug("OpenAI Whisper API error %d: %s", resp.status_code, resp.text[:200])
                return None

    return await asyncio.to_thread(_do_api_call)


async def _whisper_cli(audio_path: str, language: str) -> Optional[str]:
    """Use whisper CLI as last resort."""
    # Check if whisper CLI is available
    which_result = await asyncio.create_subprocess_exec(
        "which", "whisper",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await which_result.communicate()
    if which_result.returncode != 0:
        return None

    cmd = ["whisper", audio_path, "--output_format", "txt", "--output_dir", "/tmp"]
    if language != "auto":
        cmd.extend(["--language", language])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

    if proc.returncode == 0:
        # whisper CLI writes output to a .txt file
        txt_path = Path(audio_path).stem + ".txt"
        txt_full = os.path.join("/tmp", txt_path)
        if os.path.exists(txt_full):
            text = Path(txt_full).read_text(encoding="utf-8").strip()
            try:
                os.unlink(txt_full)
            except OSError:
                pass
            return text if text else None

    # Try parsing stdout as fallback
    output = stdout.decode("utf-8", errors="replace").strip()
    if output:
        return output

    return None
