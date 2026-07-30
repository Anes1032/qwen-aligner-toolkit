import numpy as np

from qwen_aligner_toolkit import VAD


class _Spec:
    def __init__(self, classes):
        self.classes = classes


class _Model:
    def __init__(self, classes=None):
        self.specifications = _Spec(classes) if classes else None


def _vad(classes=None):
    return VAD(model=_Model(classes), device="cpu")


def test_multilabel_output_uses_max_across_speakers():
    vad = _vad(["speaker#1", "speaker#2", "speaker#3"])
    data = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.8, 0.1, 0.0]])
    got = vad._speech_probability(data)
    assert np.allclose(got, [0.0, 0.9, 0.8])


def test_silence_is_not_reported_as_speech():
    vad = _vad(["speaker#1", "speaker#2", "speaker#3"])
    silence = np.zeros((4, 3))
    assert np.allclose(vad._speech_probability(silence), 0.0)


def test_powerset_output_uses_no_speaker_class():
    vad = _vad(["speaker#1", "speaker#2"])
    data = np.array([[0.9, 0.05, 0.03, 0.02], [0.1, 0.6, 0.2, 0.1]])
    got = vad._speech_probability(data)
    assert np.allclose(got, [1.0 - 0.9, 1.0 - 0.1])


def test_single_channel_passthrough():
    vad = _vad()
    data = np.array([[0.3], [0.7]])
    assert np.allclose(vad._speech_probability(data), [0.3, 0.7])


def test_one_dimensional_passthrough():
    vad = _vad()
    data = np.array([0.2, 0.4])
    assert np.allclose(vad._speech_probability(data), [0.2, 0.4])
