import torch

from qwen_aligner_toolkit.audio import resolve_device


def test_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_explicit_cuda_index():
    assert resolve_device("cuda:0") == torch.device("cuda:0")


def test_none_auto_detect():
    d = resolve_device(None)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert d.type == expected


def test_torch_device_passthrough():
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")
