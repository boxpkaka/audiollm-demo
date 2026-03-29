import re
from typing import Any, TypedDict

from .config import (
    ASR_REQUEST_TIMEOUT,
    SECONDARY_VLLM_BASE_URL,
    SECONDARY_VLLM_MODEL_NAME,
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
)
from .http_client import get_client


class ASRResult(TypedDict):
    transcription: str
    reported_hotwords: list[str]
    raw_text: str


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(content or "")


def build_prompt(hotwords: list[str]) -> str:
    hw_str = ",".join(hotwords) if hotwords else ""
    return f"Hotwords:{hw_str}\nTranscribe the following audio:"


def build_single_turn_messages(prompt_text: str, audio_wav_base64: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_wav_base64,
                        "format": "wav",
                    },
                },
            ],
        }
    ]


def _parse_hotwords_field(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null", "-"}:
        return []
    return [item.strip() for item in re.split(r"[,，;；]", text) if item.strip()]


def _postprocess_asr_text(text: str) -> str:
    """Normalize provider-specific wrappers to plain transcription text."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    # Qwen3-ASR: "language Chinese<asr_text>..."
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*<asr_text>\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*[:：-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def parse_model_output(raw_text: str) -> ASRResult:
    """Parse model output containing ``Transcription:`` / ``Hotwords:`` lines."""
    raw = str(raw_text or "").strip()
    if not raw:
        return ASRResult(transcription="", reported_hotwords=[], raw_text="")

    normalized = raw.replace("\\r\\n", "\n").replace("\\n", "\n")

    transcription_match = re.search(
        r"(?:^|\n)\s*transcription\s*:\s*(.+?)(?=\n\s*hotwords\s*:|\Z)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    hotwords_match = re.search(
        r"(?:^|\n)\s*hotwords\s*:\s*(.+?)(?=\n\s*[A-Za-z_]+\s*:|\Z)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )

    transcription = (
        transcription_match.group(1).strip() if transcription_match else normalized.strip()
    )
    transcription = _postprocess_asr_text(transcription)
    reported_hotwords = (
        _parse_hotwords_field(hotwords_match.group(1)) if hotwords_match else []
    )

    return ASRResult(
        transcription=transcription,
        reported_hotwords=reported_hotwords,
        raw_text=raw,
    )


async def _query_audio_model_by_endpoint(
    audio_wav_base64: str,
    *,
    base_url: str,
    model_name: str,
    hotwords: list[str] | None,
) -> ASRResult:
    client = get_client()
    prompt_text = build_prompt(hotwords or [])
    base = base_url.rstrip("/")

    resp = await client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": build_single_turn_messages(prompt_text, audio_wav_base64),
            "max_tokens": 512,
        },
        timeout=ASR_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    raw_text = _content_to_text(resp.json()["choices"][0]["message"]["content"])
    return parse_model_output(raw_text)


async def query_audio_model(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
) -> ASRResult:
    return await _query_audio_model_by_endpoint(
        audio_wav_base64,
        base_url=VLLM_BASE_URL,
        model_name=VLLM_MODEL_NAME,
        hotwords=hotwords,
    )


async def query_audio_model_secondary(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
) -> ASRResult:
    return await _query_audio_model_by_endpoint(
        audio_wav_base64,
        base_url=SECONDARY_VLLM_BASE_URL,
        model_name=SECONDARY_VLLM_MODEL_NAME,
        hotwords=hotwords,
    )
