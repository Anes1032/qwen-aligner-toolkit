import io

import numpy as np
import pytest
import soundfile as sf

from qwen_aligner_toolkit.audio import load_audio, slice_audio


def test_load_audio_np_tuple_no_resample():
    sr = 16000
    arr = np.zeros(sr * 2, dtype=np.float32)
    wav, out_sr = load_audio((arr, sr))
    assert wav.shape == (32000,)
    assert out_sr == 16000
    assert wav.dtype == np.float32


def test_load_audio_resample_8k_to_16k():
    arr = np.zeros(8000, dtype=np.float32)
    wav, sr = load_audio((arr, 8000))
    assert wav.shape == (16000,)
    assert sr == 16000


def test_load_audio_resample_48k_to_16k():
    arr = np.zeros(48000, dtype=np.float32)
    wav, sr = load_audio((arr, 48000))
    assert wav.shape == (16000,)
    assert sr == 16000


def test_load_audio_bytes():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(32000, dtype=np.float32), 16000, format="WAV")
    wav, sr = load_audio(buf.getvalue())
    assert wav.shape == (32000,)
    assert sr == 16000


def test_load_audio_path(tmp_path):
    path = tmp_path / "test.wav"
    sf.write(str(path), np.zeros(24000, dtype=np.float32), 24000, format="WAV")
    wav, sr = load_audio(str(path))
    assert wav.shape == (16000,)
    assert sr == 16000


def test_load_audio_stereo_downmix():
    sr = 16000
    stereo = np.zeros((sr, 2), dtype=np.float32)
    stereo[:, 0] = 1.0
    stereo[:, 1] = -1.0
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    wav, _ = load_audio(buf.getvalue())
    assert wav.ndim == 1
    assert wav.shape == (sr,)
    assert np.allclose(wav, 0.0, atol=1e-3)


def test_load_audio_unsupported_type():
    with pytest.raises(TypeError):
        load_audio(12345)


def test_slice_audio_basic():
    wav = np.arange(32000, dtype=np.float32)
    sliced = slice_audio(wav, 16000, 0.5, 1.5)
    assert sliced.shape == (16000,)


def test_slice_audio_with_padding():
    wav = np.arange(32000, dtype=np.float32)
    sliced = slice_audio(wav, 16000, 0.5, 1.5, padding_sec=0.1)
    assert sliced.shape == (int(16000 * 1.2),)


def test_slice_audio_clamped_at_bounds():
    wav = np.arange(16000, dtype=np.float32)
    sliced = slice_audio(wav, 16000, 0.0, 2.0, padding_sec=1.0)
    assert sliced.shape == (16000,)
