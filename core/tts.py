"""
NAOMI Agent - Text-to-Speech Engine
Uses edge-tts (free Microsoft Edge TTS) for high-quality Chinese/English voice.
Falls back to gTTS if edge-tts unavailable.
"""
import asyncio
import os
import time
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("naomi.tts")

TTS_DIR = Path("/tmp/naomi_tts")
TTS_DIR.mkdir(exist_ok=True)

# Default voice: zh-TW female (natural, warm)
DEFAULT_VOICE = "zh-TW-HsiaoChenNeural"
# Alternative voices:
# zh-TW-YunJheNeural (male), zh-CN-XiaoxiaoNeural (simplified CN female)
# en-US-AriaNeural (English female)


async def text_to_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Convert text to OGG audio file using edge-tts.
    Returns path to the audio file, or None on failure."""
    if not text or len(text.strip()) == 0:
        return None

    if output_path is None:
        output_path = str(TTS_DIR / f"tts_{int(time.time())}_{os.getpid()}.ogg")

    # Truncate very long text
    if len(text) > 2000:
        text = text[:2000] + "..."

    try:
        return await _edge_tts(text, voice, output_path)
    except Exception as e:
        logger.warning("edge-tts failed: %s, trying gTTS fallback", e)
        try:
            return await _gtts_fallback(text, output_path)
        except Exception as e2:
            logger.error("All TTS backends failed: %s", e2)
            return None


async def _edge_tts(text: str, voice: str, output_path: str) -> str:
    """Use edge-tts library for high-quality synthesis."""
    try:
        import edge_tts
    except ImportError:
        logger.info("Installing edge-tts...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "edge-tts"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import edge_tts

    # edge-tts outputs MP3 by default — convert to OGG for Telegram
    mp3_path = output_path.replace(".ogg", ".mp3")

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(mp3_path)

    # Convert MP3 → OGG using ffmpeg (Telegram voice needs OGG/Opus)
    ogg_path = output_path
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", mp3_path,
        "-c:a", "libopus", "-b:a", "48k",
        ogg_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

    # Cleanup MP3
    try:
        os.unlink(mp3_path)
    except OSError:
        pass

    if os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 0:
        return ogg_path

    raise RuntimeError("ffmpeg conversion failed")


async def _gtts_fallback(text: str, output_path: str) -> str:
    """Fallback: gTTS (Google Translate TTS)."""
    try:
        from gtts import gTTS
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gTTS"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from gtts import gTTS

    mp3_path = output_path.replace(".ogg", ".mp3")
    # Detect language: if mostly CJK → zh-TW, else en
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    lang = "zh-TW" if cjk_count > len(text) * 0.3 else "en"

    tts = gTTS(text=text, lang=lang)
    tts.save(mp3_path)

    # Convert to OGG
    ogg_path = output_path
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", mp3_path,
        "-c:a", "libopus", "-b:a", "48k",
        ogg_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

    try:
        os.unlink(mp3_path)
    except OSError:
        pass

    if os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 0:
        return ogg_path

    raise RuntimeError("gTTS + ffmpeg conversion failed")


def cleanup_old_files(max_age_hours: int = 24):
    """Remove TTS files older than max_age_hours."""
    cutoff = time.time() - max_age_hours * 3600
    for f in TTS_DIR.iterdir():
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except OSError:
                pass
