from __future__ import annotations

import logging
import math
import threading

import numpy as np
import torch

from .audio import AudioInput, load_audio, resolve_device

logger = logging.getLogger(__name__)


DEFAULT_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"


class VAD:
    def __init__(self, model, device: torch.device):
        self._model = model
        self._inference = None
        self._device = device
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        hf_token: str,
        model_id: str = DEFAULT_SEGMENTATION_MODEL,
        device: str | torch.device | None = None,
    ) -> VAD:
        from pyannote.audio import Model
        device_obj = resolve_device(device)
        model = Model.from_pretrained(model_id, token=hf_token)
        if device_obj.type == "cuda" and torch.cuda.is_available():
            model = model.to(device_obj)
        return cls(model=model, device=device_obj)

    def detect(
        self,
        audio: AudioInput,
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration_on: float = 0.25,
        min_duration_off: float = 0.5,
    ) -> list[dict]:
        waveform, sample_rate = load_audio(audio)
        if len(waveform) == 0:
            return []
        emissions, frame_times = self._infer(waveform, sample_rate)
        return _binarize(
            emissions, frame_times,
            onset=onset, offset=offset,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

    def _infer(self, waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        from pyannote.audio import Inference
        with self._lock:
            if self._inference is None:
                self._inference = Inference(self._model)
        audio_input = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": sample_rate,
        }
        with torch.no_grad():
            out = self._inference(audio_input)
        data = np.asarray(out.data)
        if data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])
        if data.ndim == 2 and data.shape[-1] >= 2:
            speech_prob = 1.0 - data[:, 0]
        elif data.ndim == 2:
            speech_prob = data[:, 0]
        else:
            speech_prob = data
        speech_prob = np.ascontiguousarray(np.asarray(speech_prob).ravel(), dtype=np.float32)
        n_frames = len(speech_prob)
        if n_frames == 0:
            return speech_prob, np.array([], dtype=np.float64)
        duration_sec = float(len(waveform)) / float(sample_rate)
        frame_times = (np.arange(n_frames, dtype=np.float64) + 0.5) * (duration_sec / n_frames)
        return speech_prob, frame_times

    def release(self) -> None:
        import gc
        with self._lock:
            model = self._model
            self._model = None
            self._inference = None
        if model is not None and hasattr(model, "to"):
            try:
                model.to(torch.device("cpu"))
            except Exception:
                pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _binarize(
    speech_prob: np.ndarray,
    frame_times: np.ndarray,
    onset: float,
    offset: float,
    min_duration_on: float,
    min_duration_off: float,
) -> list[dict]:
    if len(speech_prob) == 0:
        return []
    in_speech = False
    speech_start = 0.0
    segments: list[dict] = []
    for i in range(len(speech_prob)):
        p = float(speech_prob[i])
        t = float(frame_times[i])
        if not in_speech and p >= onset:
            speech_start = t
            in_speech = True
        elif in_speech and p < offset:
            segments.append({"start": speech_start, "end": t})
            in_speech = False
    if in_speech:
        segments.append({"start": speech_start, "end": float(frame_times[-1])})

    merged: list[dict] = []
    for s in segments:
        if merged and s["start"] - merged[-1]["end"] < min_duration_off:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    return [s for s in merged if s["end"] - s["start"] >= min_duration_on]


def merge_segments(
    segments: list[dict],
    max_chunk_sec: float = 30.0,
    max_gap_sec: float = 0.5,
) -> list[dict]:
    if not segments:
        return []
    merged = [{"start": segments[0]["start"], "end": segments[0]["end"]}]
    for s in segments[1:]:
        prev = merged[-1]
        gap = s["start"] - prev["end"]
        combined = s["end"] - prev["start"]
        if gap <= max_gap_sec and combined <= max_chunk_sec:
            prev["end"] = s["end"]
        else:
            merged.append({"start": s["start"], "end": s["end"]})

    result: list[dict] = []
    for s in merged:
        duration = s["end"] - s["start"]
        if duration <= max_chunk_sec:
            result.append(s)
            continue
        n_pieces = max(1, math.ceil(duration / max_chunk_sec))
        piece = duration / n_pieces
        for i in range(n_pieces):
            result.append({
                "start": s["start"] + i * piece,
                "end": s["start"] + (i + 1) * piece,
            })
    return result
