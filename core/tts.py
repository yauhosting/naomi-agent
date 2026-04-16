"""
NAOMI Agent - Text-to-Speech Engine

Backend priority (configurable via config.yaml tts.backend):
  - "kokoro"  : Local MLX Kokoro (fast, 54 preset voices, no cloning)
  - "qwen3"   : Local MLX Qwen3-TTS (voice cloning, emotion control)
  - "edge"    : Cloud-based Microsoft Edge TTS
  - "auto"    : Try kokoro → qwen3 → edge → gTTS in order

All backends output OGG/Opus files compatible with Telegram voice messages.
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

import yaml

logger = logging.getLogger("naomi.tts")

TTS_DIR = Path("/tmp/naomi_tts")
TTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_config_cache: Optional[dict] = None


def _load_tts_config() -> dict:
    """Load TTS config from config.yaml, with defaults."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    defaults: dict = {
        "backend": "auto",
        "kokoro": {
            "model": "mlx-community/Kokoro-82M-bf16",
            "voice": "zf_xiaobei",
            "speed": 1.0,
            "lang_code": "z",
        },
        "qwen3": {
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
            "speaker": "Vivian",
            "instruct": "",
            "ref_audio": "",
            "ref_text": "",
        },
        "edge": {
            "voice": "zh-TW-HsiaoChenNeural",
        },
    }

    config_path = Path(__file__).parent.parent / "config.yaml"
    try:
        with open(config_path, "r") as f:
            full = yaml.safe_load(f)
        if isinstance(full, dict) and "tts" in full:
            tts = full["tts"]
            defaults["backend"] = tts.get("backend", defaults["backend"])
            for section in ("kokoro", "qwen3", "edge"):
                if section in tts and isinstance(tts[section], dict):
                    defaults[section].update(tts[section])
    except Exception as e:
        logger.warning("Failed to load TTS config: %s — using defaults", e)

    _config_cache = defaults
    return defaults


def reload_config() -> None:
    """Force reload TTS config (e.g. after config.yaml changes)."""
    global _config_cache
    _config_cache = None


# ---------------------------------------------------------------------------
# MLX model singletons (lazy-loaded, thread-safe)
# ---------------------------------------------------------------------------
_kokoro_model = None
_kokoro_lock = threading.Lock()
_qwen3_model = None
_qwen3_lock = threading.Lock()


def _get_kokoro():
    """Lazy-load Kokoro model (thread-safe)."""
    global _kokoro_model
    if _kokoro_model is not None:
        return _kokoro_model

    with _kokoro_lock:
        # Double-check after acquiring lock
        if _kokoro_model is not None:
            return _kokoro_model

        from mlx_audio.tts.utils import load_model

        cfg = _load_tts_config()["kokoro"]
        logger.info("Loading Kokoro model: %s", cfg["model"])
        _kokoro_model = load_model(cfg["model"])
        logger.info("Kokoro model loaded")
        return _kokoro_model


def _get_qwen3():
    """Lazy-load Qwen3-TTS model (thread-safe)."""
    global _qwen3_model
    if _qwen3_model is not None:
        return _qwen3_model

    with _qwen3_lock:
        if _qwen3_model is not None:
            return _qwen3_model

        from mlx_audio.tts.utils import load_model

        cfg = _load_tts_config()["qwen3"]
        logger.info("Loading Qwen3-TTS model: %s", cfg["model"])
        _qwen3_model = load_model(cfg["model"])
        logger.info("Qwen3-TTS model loaded")
        return _qwen3_model


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

def _detect_lang_code(text: str) -> str:
    """Auto-detect Kokoro language code from text content."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    jp = sum(1 for c in text if "\u3040" <= c <= "\u30ff" or "\u31f0" <= c <= "\u31ff")

    total = len(text)
    if total == 0:
        return "z"

    if jp / total > 0.1:
        return "j"
    if cjk / total > 0.2:
        return "z"
    return "a"


def _unique_path(suffix: str = ".ogg") -> str:
    """Generate a unique temp file path (nanosecond precision)."""
    return str(TTS_DIR / f"tts_{time.time_ns()}_{os.getpid()}{suffix}")


# ---------------------------------------------------------------------------
# WAV → OGG conversion
# ---------------------------------------------------------------------------

async def _wav_to_ogg(wav_path: str, ogg_path: str) -> bool:
    """Convert WAV to OGG/Opus via ffmpeg for Telegram compatibility."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", wav_path,
        "-c:a", "libopus", "-b:a", "48k",
        ogg_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

    try:
        os.unlink(wav_path)
    except OSError:
        pass

    return os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 0


def _cleanup_file(path: str) -> None:
    """Silently remove a file if it exists."""
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Backend: Kokoro (MLX)
# ---------------------------------------------------------------------------

async def _kokoro_tts(text: str, output_path: str) -> str:
    """Generate speech using Kokoro via mlx-audio."""
    cfg = _load_tts_config()["kokoro"]
    model = await asyncio.to_thread(_get_kokoro)

    lang_code = cfg.get("lang_code", "auto")
    if lang_code == "auto":
        lang_code = _detect_lang_code(text)

    voice = cfg.get("voice", "zf_xiaobei")
    speed = cfg.get("speed", 1.0)

    def _generate():
        import numpy as np
        segments = []
        sample_rate = 24000
        for result in model.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
        ):
            segments.append(np.asarray(result.audio))
            sample_rate = result.sample_rate

        if not segments:
            return None, 0
        return np.concatenate(segments), sample_rate

    audio_np, sr = await asyncio.to_thread(_generate)
    if audio_np is None:
        raise RuntimeError("Kokoro generated no audio")

    wav_path = output_path.replace(".ogg", ".wav")
    try:
        import soundfile as sf
        await asyncio.to_thread(sf.write, wav_path, audio_np, sr)
        if await _wav_to_ogg(wav_path, output_path):
            return output_path
        raise RuntimeError("Kokoro: WAV→OGG conversion failed")
    finally:
        _cleanup_file(wav_path)


# ---------------------------------------------------------------------------
# Backend: Qwen3-TTS (MLX)
# ---------------------------------------------------------------------------

async def _qwen3_tts(text: str, output_path: str) -> str:
    """Generate speech using Qwen3-TTS via mlx-audio."""
    cfg = _load_tts_config()["qwen3"]
    model = await asyncio.to_thread(_get_qwen3)

    ref_audio = cfg.get("ref_audio", "")
    ref_text = cfg.get("ref_text", "")
    speaker = cfg.get("speaker", "Vivian")
    instruct = cfg.get("instruct", "")

    def _generate():
        import numpy as np
        segments = []
        sample_rate = 24000

        kwargs: dict = {"text": text}

        if ref_audio and os.path.isfile(ref_audio):
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
        else:
            kwargs["voice"] = speaker
            if instruct:
                kwargs["instruct"] = instruct

        for result in model.generate(**kwargs):
            segments.append(np.asarray(result.audio))
            sample_rate = result.sample_rate

        if not segments:
            return None, 0
        return np.concatenate(segments), sample_rate

    audio_np, sr = await asyncio.to_thread(_generate)
    if audio_np is None:
        raise RuntimeError("Qwen3-TTS generated no audio")

    wav_path = output_path.replace(".ogg", ".wav")
    try:
        import soundfile as sf
        await asyncio.to_thread(sf.write, wav_path, audio_np, sr)
        if await _wav_to_ogg(wav_path, output_path):
            return output_path
        raise RuntimeError("Qwen3-TTS: WAV→OGG conversion failed")
    finally:
        _cleanup_file(wav_path)


# ---------------------------------------------------------------------------
# Backend: Edge TTS (cloud)
# ---------------------------------------------------------------------------

async def _edge_tts(text: str, output_path: str) -> str:
    """Use edge-tts library for cloud-based synthesis."""
    try:
        import edge_tts
    except ImportError:
        logger.info("Installing edge-tts...")
        await asyncio.to_thread(
            lambda: subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "edge-tts"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        )
        import edge_tts

    cfg = _load_tts_config()["edge"]
    voice = cfg.get("voice", "zh-TW-HsiaoChenNeural")

    mp3_path = output_path.replace(".ogg", ".mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(mp3_path)

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", mp3_path,
        "-c:a", "libopus", "-b:a", "48k",
        output_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

    _cleanup_file(mp3_path)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    raise RuntimeError("edge-tts: ffmpeg conversion failed")


# ---------------------------------------------------------------------------
# Backend: gTTS (last resort)
# ---------------------------------------------------------------------------

async def _gtts_fallback(text: str, output_path: str) -> str:
    """Fallback: gTTS (Google Translate TTS)."""
    try:
        from gtts import gTTS
    except ImportError:
        await asyncio.to_thread(
            lambda: subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "gTTS"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        )
        from gtts import gTTS

    mp3_path = output_path.replace(".ogg", ".mp3")
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    lang = "zh-TW" if cjk_count > len(text) * 0.3 else "en"

    tts_obj = gTTS(text=text, lang=lang)
    await asyncio.to_thread(tts_obj.save, mp3_path)

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", mp3_path,
        "-c:a", "libopus", "-b:a", "48k",
        output_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

    _cleanup_file(mp3_path)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    raise RuntimeError("gTTS + ffmpeg conversion failed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BACKENDS = {
    "kokoro": _kokoro_tts,
    "qwen3": _qwen3_tts,
    "edge": _edge_tts,
}

_AUTO_ORDER = ["kokoro", "qwen3", "edge"]


async def text_to_speech(
    text: str,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Convert text to OGG audio file.

    Args:
        text: Text to synthesize.
        output_path: Output file path. Auto-generated if None.

    Returns:
        Path to the OGG audio file, or None on failure.
    """
    if not text or len(text.strip()) == 0:
        return None

    if output_path is None:
        output_path = _unique_path(".ogg")

    # Truncate very long text
    if len(text) > 2000:
        text = text[:2000] + "..."

    cfg = _load_tts_config()
    backend = cfg["backend"]

    if backend in _BACKENDS:
        try:
            return await _BACKENDS[backend](text, output_path)
        except Exception as e:
            logger.warning("%s TTS failed: %s — falling back", backend, e)
            try:
                return await _edge_tts(text, output_path)
            except Exception:
                pass
            try:
                return await _gtts_fallback(text, output_path)
            except Exception as e2:
                logger.error("All TTS backends failed: %s", e2)
                return None
    else:
        for name in _AUTO_ORDER:
            try:
                return await _BACKENDS[name](text, output_path)
            except Exception as e:
                logger.warning("%s TTS failed: %s — trying next", name, e)
                continue

        try:
            return await _gtts_fallback(text, output_path)
        except Exception as e:
            logger.error("All TTS backends failed: %s", e)
            return None


def cleanup_old_files(max_age_hours: int = 24) -> None:
    """Remove TTS files older than max_age_hours."""
    cutoff = time.time() - max_age_hours * 3600
    for f in TTS_DIR.iterdir():
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except OSError:
                pass
