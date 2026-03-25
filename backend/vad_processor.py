import logging

import numpy as np

try:
    from ten_vad import TenVad
except Exception:  # pragma: no cover - depends on optional native runtime
    TenVad = None

from .config import (
    HOP_SIZE,
    SILENCE_DURATION_MS,
    VAD_END_FRAMES,
    VAD_SMOOTHING_ALPHA,
    VAD_START_FRAMES,
    VAD_THRESHOLD,
)


logger = logging.getLogger(__name__)


class _EnergyVad:
    """Simple RMS-energy fallback VAD returning pseudo-probability [0, 1]."""

    def __init__(self, floor: float = 0.008, ceil: float = 0.06):
        self.floor = max(1e-6, floor)
        self.ceil = max(self.floor + 1e-6, ceil)

    def process(self, pcm_frame: np.ndarray) -> float:
        energy = float(np.sqrt(np.mean(np.square(pcm_frame), dtype=np.float32)))
        normalized = (energy - self.floor) / (self.ceil - self.floor)
        return float(min(1.0, max(0.0, normalized)))


def _patch_tenvad_destructor():
    """Guard against noisy AttributeError in ten-vad __del__."""
    if TenVad is None:
        return

    original_del = getattr(TenVad, "__del__", None)
    if not callable(original_del):
        return

    def _safe_del(self):
        # ten-vad may create partially initialized objects; skip unsafe cleanup.
        if not hasattr(self, "vad_library"):
            return
        try:
            original_del(self)
        except AttributeError:
            pass

    TenVad.__del__ = _safe_del


_patch_tenvad_destructor()


class VADProcessor:
    def __init__(
        self,
        hop_size: int = HOP_SIZE,
        threshold: float = VAD_THRESHOLD,
        silence_duration_ms: int = SILENCE_DURATION_MS,
        smoothing_alpha: float = VAD_SMOOTHING_ALPHA,
        start_frames: int = VAD_START_FRAMES,
        end_frames: int = VAD_END_FRAMES,
    ):
        self.vad = self._create_vad_backend()
        self.hop_size = hop_size
        self.threshold = threshold
        self.silence_frames = silence_duration_ms // 10
        self.end_frames = max(1, end_frames)
        self.start_frames = max(1, start_frames)
        self.smoothing_alpha = min(1.0, max(0.0, smoothing_alpha))
        self.audio_buffer: list[np.ndarray] = []
        self.pre_speech_buffer: list[np.ndarray] = []
        self.silent_count = 0
        self.speech_count = 0
        self.is_speaking = False
        self.smoothed_prob: float | None = None

    def _create_vad_backend(self):
        if TenVad is None:
            logger.warning(
                "ten-vad is unavailable; using fallback energy VAD. "
                "Install ten-vad and system libc++ (e.g. apt install libc++1)."
            )
            return _EnergyVad()

        try:
            return TenVad()
        except OSError as exc:
            logger.warning(
                "TEN VAD native library failed to load (%s). "
                "Using fallback energy VAD. "
                "Install system libc++ (e.g. apt install libc++1).",
                exc,
            )
            return _EnergyVad()

    def process(self, pcm_frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame (hop_size samples, float32).
        Returns the full speech segment when speech-to-silence transition
        is detected, otherwise None.
        """
        raw_prob = self.vad.process(pcm_frame)
        if self.smoothed_prob is None:
            self.smoothed_prob = raw_prob
        else:
            a = self.smoothing_alpha
            self.smoothed_prob = (a * self.smoothed_prob) + ((1.0 - a) * raw_prob)

        is_speech = self.smoothed_prob > self.threshold
        frame_copy = pcm_frame.copy()

        if not self.is_speaking:
            self.pre_speech_buffer.append(frame_copy)
            if len(self.pre_speech_buffer) > self.start_frames:
                del self.pre_speech_buffer[0]

            if is_speech:
                self.speech_count += 1
            else:
                self.speech_count = 0

            if self.speech_count >= self.start_frames:
                self.is_speaking = True
                self.silent_count = 0
                self.audio_buffer.extend(self.pre_speech_buffer)
                self.pre_speech_buffer.clear()
            return None

        # Speaking state.
        self.audio_buffer.append(frame_copy)
        if is_speech:
            self.silent_count = 0
        else:
            self.silent_count += 1
            end_threshold = max(self.silence_frames, self.end_frames)
            if self.silent_count >= end_threshold:
                # Trim trailing silence (keep a small tail for natural sound)
                keep_tail = min(10, end_threshold)
                trim = end_threshold - keep_tail
                if trim > 0:
                    del self.audio_buffer[-trim:]
                segment = np.concatenate(self.audio_buffer)
                self._reset()
                return segment

        return None

    def flush(self) -> np.ndarray | None:
        """Flush any remaining buffered speech (e.g. on disconnect)."""
        if self.audio_buffer and self.is_speaking:
            segment = np.concatenate(self.audio_buffer)
            self._reset()
            return segment
        self._reset()
        return None

    def _reset(self):
        self.audio_buffer.clear()
        self.pre_speech_buffer.clear()
        self.silent_count = 0
        self.speech_count = 0
        self.is_speaking = False
        self.smoothed_prob = None
