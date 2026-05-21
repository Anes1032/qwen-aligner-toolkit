from qwen_aligner_toolkit.compat.nagisa_stub import STUB_SOURCE


def test_stub_source_runs_as_module():
    ns: dict = {}
    exec(STUB_SOURCE, ns)
    tagging = ns["tagging"]
    result = tagging("日本語テスト")
    assert result.words == ["日", "本", "語", "テ", "ス", "ト"]


def test_stub_source_empty_string():
    ns: dict = {}
    exec(STUB_SOURCE, ns)
    result = ns["tagging"]("")
    assert result.words == []
