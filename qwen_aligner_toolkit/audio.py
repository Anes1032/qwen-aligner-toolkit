import io
from pathlib import Path
from typing import Union

import numpy as np
import torch

AudioInput = Union[str, Path, bytes, np.ndarray, torch.Tensor, tuple]

TARGET_SAMPLE_RATE = 16000


def resolve_device(device: Union[str, torch.device, None]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resample(arr: np.ndarray, sr: int, target_sample_rate: int) -> np.ndarray:
    if sr == target_sample_rate:
        return arr.astype(np.float32, copy=False)
    import soxr
    return soxr.resample(arr.astype(np.float32, copy=False), sr, target_sample_rate).astype(np.float32, copy=False)


def load_audio(audio: AudioInput, target_sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    if isinstance(audio, tuple) and len(audio) == 2:
        arr, sr = audio
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
        return _resample(arr, sr, target_sample_rate), target_sample_rate

    if isinstance(audio, np.ndarray):
        return load_audio((audio, target_sample_rate), target_sample_rate)

    if isinstance(audio, torch.Tensor):
        return load_audio((audio, target_sample_rate), target_sample_rate)

    import soundfile as sf

    if isinstance(audio, (str, Path)):
        src = str(audio)
        if src.startswith(("http://", "https://")):
            from urllib.request import urlopen
            with urlopen(src) as resp:
                payload = resp.read()
            data, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
        else:
            data, sr = sf.read(src, dtype="float32", always_2d=False)
    elif isinstance(audio, bytes):
        data, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=False)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return _resample(arr, sr, target_sample_rate), target_sample_rate


def slice_audio(
    waveform: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    padding_sec: float = 0.0,
) -> np.ndarray:
    total = len(waveform)
    start = max(0, int(round((start_sec - padding_sec) * sample_rate)))
    end = min(total, int(round((end_sec + padding_sec) * sample_rate)))
    return waveform[start:end].astype(np.float32)
