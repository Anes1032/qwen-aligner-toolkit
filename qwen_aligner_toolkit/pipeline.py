from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .aligner import Aligner, Segment, Word
from .audio import AudioInput, resolve_device
from .diarizer import Diarizer
from .vad import VAD, merge_segments

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    words: list[Word]
    transcript: str

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "words": [w.to_dict() for w in self.words],
        }


class Pipeline:
    def __init__(
        self,
        aligner: Aligner,
        vad: VAD | None = None,
        diarizer: Diarizer | None = None,
    ):
        self.aligner = aligner
        self.vad = vad
        self.diarizer = diarizer

    @classmethod
    def from_pretrained(
        cls,
        hf_token: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        diarization_model: str = "pyannote/speaker-diarization-community-1",
        vad_model: str = "pyannote/segmentation-3.0",
        with_vad: bool = True,
        with_diarization: bool = True,
    ) -> Pipeline:
        device_obj = resolve_device(device)
        aligner = Aligner.from_pretrained(model_id=aligner_model, dtype=dtype, device_map=device_obj)
        vad = None
        diarizer = None
        if (with_vad or with_diarization) and hf_token is None:
            raise ValueError("hf_token required for VAD or diarization")
        if with_vad:
            vad = VAD.from_pretrained(hf_token=hf_token, model_id=vad_model, device=device_obj)
        if with_diarization:
            diarizer = Diarizer.from_pretrained(hf_token=hf_token, model_id=diarization_model, device=device_obj)
        return cls(aligner=aligner, vad=vad, diarizer=diarizer)

    def align(
        self,
        text: str,
        audio: AudioInput,
        language: str,
        diarize: bool = False,
    ) -> PipelineResult:
        words = self.aligner.align(text, audio, language)
        if diarize and self.diarizer is not None:
            words = self.diarizer.label_words(words, audio)
        return PipelineResult(words=words, transcript=text)

    def align_segments(
        self,
        segments: list[dict | Segment],
        audio: AudioInput,
        language: str,
        diarize: bool = False,
        padding_sec: float = 0.2,
    ) -> PipelineResult:
        words = self.aligner.align_segments(segments, audio, language, padding_sec=padding_sec)
        if diarize and self.diarizer is not None:
            words = self.diarizer.label_words(words, audio)
        transcript = " ".join(
            (s.text if isinstance(s, Segment) else s["text"]).strip() for s in segments
        ).strip()
        return PipelineResult(words=words, transcript=transcript)

    def vad_chunks(
        self,
        audio: AudioInput,
        max_chunk_sec: float = 30.0,
        max_gap_sec: float = 0.5,
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration_on: float = 0.25,
        min_duration_off: float = 0.5,
    ) -> list[dict]:
        if self.vad is None:
            raise RuntimeError("VAD is not initialized. Use Pipeline.from_pretrained(..., with_vad=True).")
        raw = self.vad.detect(
            audio,
            onset=onset, offset=offset,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )
        return merge_segments(raw, max_chunk_sec=max_chunk_sec, max_gap_sec=max_gap_sec)

    def release(self) -> None:
        if self.aligner is not None:
            self.aligner.release()
        if self.vad is not None:
            self.vad.release()
        if self.diarizer is not None:
            self.diarizer.release()
