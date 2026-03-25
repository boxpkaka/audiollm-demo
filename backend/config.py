import os

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Amphion/Amphion-3B")
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.6"))
SILENCE_DURATION_MS = int(os.getenv("SILENCE_DURATION_MS", "600"))
# Exponential moving average smoothing for frame-level VAD probability.
# 0 means no smoothing; closer to 1 means stronger smoothing.
VAD_SMOOTHING_ALPHA = float(os.getenv("VAD_SMOOTHING_ALPHA", "0.35"))
# Require N consecutive speech frames before entering speaking state.
VAD_START_FRAMES = int(os.getenv("VAD_START_FRAMES", "3"))
# Require N consecutive non-speech frames before ending speaking state.
# Keep backward compatibility by defaulting to SILENCE_DURATION_MS / 10.
VAD_END_FRAMES = int(os.getenv("VAD_END_FRAMES", str(max(1, SILENCE_DURATION_MS // 10))))
HOP_SIZE = 160  # 10ms at 16kHz, TEN VAD recommended
SAMPLE_RATE = 16000
