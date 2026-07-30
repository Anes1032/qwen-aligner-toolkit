from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import torch

from .aligner import Word
from .audio import AudioInput, load_audio, resolve_device
from .hub import load_from_cache_or_hub

logger = logging.getLogger(__name__)


DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class Diarizer:
    def __init__(self, pipeline, device: torch.device):
        self._pipeline = pipeline
        self._device = device
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        hf_token: str,
        model_id: str = DEFAULT_DIARIZATION_MODEL,
        device: str | torch.device | None = None,
    ) -> Diarizer:
        from pyannote.audio import Pipeline
        device_obj = resolve_device(device)
        pipeline = load_from_cache_or_hub(
            lambda: Pipeline.from_pretrained(model_id, token=hf_token),
            f"pyannote pipeline {model_id}",
        )
        if device_obj.type == "cuda" and torch.cuda.is_available():
            pipeline.to(device_obj)
        return cls(pipeline=pipeline, device=device_obj)

    def diarize(self, audio: AudioInput, sample_rate: int | None = None) -> list[SpeakerTurn]:
        waveform_np, sr = load_audio(audio)
        if sample_rate is not None and sample_rate != sr:
            sr = sample_rate
        try:
            output = self._pipeline({
                "waveform": torch.from_numpy(waveform_np).unsqueeze(0),
                "sample_rate": sr,
            })
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
            return []
        annotation = _extract_annotation(output)
        if annotation is None:
            return []
        return [
            SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=str(label))
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]

    def label_words(
        self,
        words: list[Word],
        audio: AudioInput,
        fallback_nearest: bool = True,
        min_duration_sec: float = 0.3,
    ) -> list[Word]:
        if not words:
            return []
        turns = self.diarize(audio)
        if not turns:
            return words

        labelled: list[Word] = []
        for w in words:
            spk = _dominant_speaker(w.start_time, w.end_time, turns, fallback_nearest=fallback_nearest)
            labelled.append(Word(text=w.text, start_time=w.start_time, end_time=w.end_time, speaker=spk))
        return _smooth_speaker_runs(labelled, min_duration_sec=min_duration_sec)

    @property
    def segmentation_model(self):
        pipeline = self._pipeline
        if pipeline is None:
            return None
        for attr in ("_segmentation", "segmentation_model", "segmentation"):
            obj = getattr(pipeline, attr, None)
            if obj is None:
                continue
            inner = getattr(obj, "model", obj)
            if hasattr(inner, "to") and callable(inner.to):
                return inner
        return None

    @staticmethod
    def dominant_speaker(
        start: float,
        end: float,
        turns: list[SpeakerTurn],
        fallback_nearest: bool = False,
    ) -> str | None:
        return _dominant_speaker(start, end, turns, fallback_nearest=fallback_nearest)

    @staticmethod
    def split_words_by_speaker(
        words: list[dict],
        *,
        min_duration_sec: float = 0.3,
        word_key: str = "word",
        start_key: str = "start",
        end_key: str = "end",
        speaker_key: str = "speaker",
        join_separator: str = " ",
    ) -> list[dict]:
        return _split_words_by_speaker(
            words,
            min_duration_sec=min_duration_sec,
            word_key=word_key,
            start_key=start_key,
            end_key=end_key,
            speaker_key=speaker_key,
            join_separator=join_separator,
        )

    def release(self) -> None:
        import gc
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
        if pipeline is not None:
            try:
                pipeline.to(torch.device("cpu"))
            except Exception:
                pass
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _extract_annotation(output):
    if hasattr(output, "itertracks"):
        return output
    for attr in ("speaker_diarization", "diarization", "annotation"):
        obj = getattr(output, attr, None)
        if obj is not None and hasattr(obj, "itertracks"):
            return obj
    return None


def _dominant_speaker(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
    fallback_nearest: bool = False,
) -> str | None:
    if not turns or end < start:
        return None
    is_point = end == start
    overlaps: dict[str, float] = {}
    for t in turns:
        ov = max(0.0, min(end, t.end) - max(start, t.start))
        if ov > 0:
            overlaps[t.speaker] = overlaps.get(t.speaker, 0.0) + ov
        elif is_point and t.start <= start <= t.end:
            overlaps.setdefault(t.speaker, 0.0)
    if overlaps:
        return max(overlaps, key=overlaps.get)
    if not fallback_nearest:
        return None
    best_spk: str | None = None
    best_dist = float("inf")
    for t in turns:
        if t.end < start:
            dist = start - t.end
        elif t.start > end:
            dist = t.start - end
        else:
            dist = 0.0
        if dist < best_dist:
            best_dist = dist
            best_spk = t.speaker
    return best_spk


def _smooth_speaker_runs(words: list[Word], min_duration_sec: float) -> list[Word]:
    if not words:
        return []
    groups: list[dict] = []
    for w in words:
        if groups and groups[-1]["speaker"] == w.speaker:
            groups[-1]["words"].append(w)
        else:
            groups.append({"speaker": w.speaker, "words": [w]})
    for g in groups:
        g["start"] = g["words"][0].start_time
        g["end"] = g["words"][-1].end_time

    def _merge_into(idx: int, target_offset: int) -> None:
        src = groups.pop(idx)
        target = idx + target_offset
        if target_offset > 0:
            target -= 1
        t = groups[target]
        if target < idx:
            t["words"].extend(src["words"])
        else:
            t["words"] = src["words"] + t["words"]
        t["start"] = t["words"][0].start_time
        t["end"] = t["words"][-1].end_time
        i = max(target - 1, 0)
        while i < len(groups) - 1:
            if groups[i]["speaker"] == groups[i + 1]["speaker"]:
                a, b = groups[i], groups[i + 1]
                a["words"].extend(b["words"])
                a["start"] = a["words"][0].start_time
                a["end"] = a["words"][-1].end_time
                del groups[i + 1]
            else:
                i += 1

    while len(groups) > 1:
        idx = None
        for i, g in enumerate(groups):
            if g["speaker"] is not None:
                continue
            prev_real = i > 0 and groups[i - 1]["speaker"] is not None
            next_real = i < len(groups) - 1 and groups[i + 1]["speaker"] is not None
            if prev_real or next_real:
                idx = i
                break
        if idx is None:
            break
        prev_dur = -1.0
        if idx > 0 and groups[idx - 1]["speaker"] is not None:
            prev_dur = groups[idx - 1]["end"] - groups[idx - 1]["start"]
        next_dur = -1.0
        if idx < len(groups) - 1 and groups[idx + 1]["speaker"] is not None:
            next_dur = groups[idx + 1]["end"] - groups[idx + 1]["start"]
        _merge_into(idx, -1 if prev_dur >= next_dur else 1)

    while len(groups) > 1:
        idx = None
        shortest = min_duration_sec
        for i, g in enumerate(groups):
            d = g["end"] - g["start"]
            if d < shortest:
                shortest = d
                idx = i
        if idx is None:
            break
        prev_dur = (groups[idx - 1]["end"] - groups[idx - 1]["start"]) if idx > 0 else -1.0
        next_dur = (groups[idx + 1]["end"] - groups[idx + 1]["start"]) if idx < len(groups) - 1 else -1.0
        _merge_into(idx, -1 if prev_dur >= next_dur else 1)

    result: list[Word] = []
    for g in groups:
        spk = g["speaker"]
        for w in g["words"]:
            result.append(Word(text=w.text, start_time=w.start_time, end_time=w.end_time, speaker=spk))
    return result


def _split_words_by_speaker(
    words: list[dict],
    *,
    min_duration_sec: float,
    word_key: str,
    start_key: str,
    end_key: str,
    speaker_key: str,
    join_separator: str,
) -> list[dict]:
    usable = [w for w in words if w.get(start_key) is not None and w.get(end_key) is not None]
    if not usable:
        return []

    def _bounds(ws: list[dict]) -> tuple[float, float]:
        starts = [float(w[start_key]) for w in ws]
        ends = [float(w[end_key]) for w in ws]
        return min(starts), max(ends)

    groups: list[dict] = []
    for w in usable:
        sp = w.get(speaker_key)
        if groups and groups[-1]["speaker"] == sp:
            groups[-1]["words"].append(w)
        else:
            groups.append({"speaker": sp, "words": [w]})
    for g in groups:
        g["start"], g["end"] = _bounds(g["words"])

    def _merge_into(idx: int, target_offset: int) -> None:
        src = groups.pop(idx)
        target = idx + target_offset
        if target_offset > 0:
            target -= 1
        t = groups[target]
        if target < idx:
            t["words"].extend(src["words"])
        else:
            t["words"] = src["words"] + t["words"]
        t["start"], t["end"] = _bounds(t["words"])
        i = max(target - 1, 0)
        while i < len(groups) - 1:
            if groups[i]["speaker"] == groups[i + 1]["speaker"]:
                a, b = groups[i], groups[i + 1]
                a["words"].extend(b["words"])
                a["start"], a["end"] = _bounds(a["words"])
                del groups[i + 1]
            else:
                i += 1

    while len(groups) > 1:
        idx = None
        for i, g in enumerate(groups):
            if g["speaker"] is not None:
                continue
            prev_real = i > 0 and groups[i - 1]["speaker"] is not None
            next_real = i < len(groups) - 1 and groups[i + 1]["speaker"] is not None
            if prev_real or next_real:
                idx = i
                break
        if idx is None:
            break
        prev_dur = -1.0
        if idx > 0 and groups[idx - 1]["speaker"] is not None:
            prev_dur = groups[idx - 1]["end"] - groups[idx - 1]["start"]
        next_dur = -1.0
        if idx < len(groups) - 1 and groups[idx + 1]["speaker"] is not None:
            next_dur = groups[idx + 1]["end"] - groups[idx + 1]["start"]
        _merge_into(idx, -1 if prev_dur >= next_dur else 1)

    while len(groups) > 1:
        idx = None
        shortest = min_duration_sec
        for i, g in enumerate(groups):
            d = g["end"] - g["start"]
            if d < shortest:
                shortest = d
                idx = i
        if idx is None:
            break
        prev_dur = (groups[idx - 1]["end"] - groups[idx - 1]["start"]) if idx > 0 else -1.0
        next_dur = (groups[idx + 1]["end"] - groups[idx + 1]["start"]) if idx < len(groups) - 1 else -1.0
        _merge_into(idx, -1 if prev_dur >= next_dur else 1)

    entries: list[dict] = []
    for g in groups:
        parts = [w.get(word_key) or "" for w in g["words"]]
        text = join_separator.join(p for p in parts if p).strip()
        if not text:
            continue
        entry: dict = {
            "text": text,
            "start_time": float(g["start"]),
            "end_time": float(g["end"]),
        }
        if g["speaker"] is not None:
            entry["speaker"] = str(g["speaker"])
        entries.append(entry)
    return entries
