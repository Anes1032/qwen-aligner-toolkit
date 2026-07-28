from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .audio import AudioInput, load_audio, resolve_device, slice_audio
from .hub import load_from_cache_or_hub

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

ISO_TO_QWEN_LANG = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "th": "Thai",
}


def _normalize_language(language: str) -> str:
    if not language:
        raise ValueError("language must be specified")
    key = language.strip().lower()
    if key in ISO_TO_QWEN_LANG:
        return ISO_TO_QWEN_LANG[key]
    return language


@dataclass
class Word:
    text: str
    start_time: float
    end_time: float
    speaker: str | None = None

    def to_dict(self) -> dict:
        out = {"text": self.text, "start_time": self.start_time, "end_time": self.end_time}
        if self.speaker is not None:
            out["speaker"] = self.speaker
        return out


@dataclass
class Segment:
    text: str
    start: float
    end: float

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(text=d["text"], start=float(d["start"]), end=float(d["end"]))


class Aligner:
    def __init__(self, model):
        self._model = model

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL,
        dtype: torch.dtype | None = None,
        device_map: str | torch.device | None = None,
        **kwargs,
    ) -> Aligner:
        from qwen_asr import Qwen3ForcedAligner
        device = resolve_device(device_map)
        resolved_map = f"{device.type}:{device.index}" if device.index is not None else device.type
        resolved_dtype = dtype if dtype is not None else (
            torch.bfloat16 if device.type == "cuda" else torch.float32
        )
        model = load_from_cache_or_hub(
            lambda: Qwen3ForcedAligner.from_pretrained(
                model_id,
                dtype=resolved_dtype,
                device_map=resolved_map,
                **kwargs,
            ),
            f"Qwen3ForcedAligner {model_id}",
        )
        logger.info(f"Loaded Qwen3ForcedAligner: {model_id} on {resolved_map} ({resolved_dtype})")
        return cls(model=model)

    def align(
        self,
        text: str,
        audio: AudioInput,
        language: str,
    ) -> list[Word]:
        if not text or not text.strip():
            return []
        waveform, sample_rate = load_audio(audio)
        return self._align_array(text, waveform, sample_rate, language, offset_sec=0.0)

    def align_segments(
        self,
        segments: list[dict | Segment],
        audio: AudioInput,
        language: str,
        padding_sec: float = 0.2,
    ) -> list[Word]:
        waveform, sample_rate = load_audio(audio)
        normalized: list[Segment] = []
        for s in segments:
            normalized.append(s if isinstance(s, Segment) else Segment.from_dict(s))

        results: list[Word] = []
        for seg in normalized:
            chunk = slice_audio(waveform, sample_rate, seg.start, seg.end, padding_sec=padding_sec)
            if len(chunk) < int(sample_rate * 0.025):
                continue
            words = self._align_array(
                seg.text, chunk, sample_rate, language, offset_sec=max(0.0, seg.start - padding_sec)
            )
            results.extend(words)
        return results

    def _align_array(
        self,
        text: str,
        waveform: np.ndarray,
        sample_rate: int,
        language: str,
        offset_sec: float,
    ) -> list[Word]:
        lang_name = _normalize_language(language)
        try:
            raw = self._model.align(
                audio=(waveform, sample_rate),
                text=text,
                language=lang_name,
            )
        except Exception as e:
            logger.warning(f"Qwen aligner failed: {e}")
            return []
        if not raw:
            return []
        return [
            Word(
                text=item.text,
                start_time=float(item.start_time) + offset_sec,
                end_time=float(item.end_time) + offset_sec,
            )
            for item in raw[0]
        ]

    def release(self) -> None:
        import gc
        model = self._model
        self._model = None
        for attr in ("model", "thinker"):
            obj = getattr(model, attr, None)
            if obj is not None and hasattr(obj, "to") and callable(getattr(obj, "to")):
                try:
                    obj.to("cpu")
                except Exception:
                    pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
