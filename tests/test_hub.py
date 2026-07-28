import pytest

from qwen_aligner_toolkit.hub import forced_offline, load_from_cache_or_hub


def test_forced_offline_sets_and_restores_flags():
    import huggingface_hub.constants as hub_constants

    before = hub_constants.HF_HUB_OFFLINE
    with forced_offline():
        assert hub_constants.HF_HUB_OFFLINE is True
    assert hub_constants.HF_HUB_OFFLINE == before


def test_forced_offline_restores_on_error():
    import huggingface_hub.constants as hub_constants

    before = hub_constants.HF_HUB_OFFLINE
    with pytest.raises(RuntimeError):
        with forced_offline():
            raise RuntimeError("boom")
    assert hub_constants.HF_HUB_OFFLINE == before


def test_loads_from_cache_without_contacting_the_hub():
    import huggingface_hub.constants as hub_constants

    seen_offline = []

    def load():
        seen_offline.append(hub_constants.HF_HUB_OFFLINE)
        return "cached model"

    assert load_from_cache_or_hub(load, "thing") == "cached model"
    assert seen_offline == [True]


def test_falls_back_to_the_hub_on_cache_miss():
    import huggingface_hub.constants as hub_constants
    from huggingface_hub.errors import LocalEntryNotFoundError

    seen_offline = []

    def load():
        seen_offline.append(hub_constants.HF_HUB_OFFLINE)
        if len(seen_offline) == 1:
            raise LocalEntryNotFoundError("not cached")
        return "downloaded model"

    assert load_from_cache_or_hub(load, "thing") == "downloaded model"
    assert seen_offline == [True, False]


def test_hub_error_propagates_when_cache_misses_and_hub_is_unreachable():
    from huggingface_hub.errors import LocalEntryNotFoundError
    from requests.exceptions import ConnectionError as RequestsConnectionError

    def load():
        if load.calls == 0:
            load.calls += 1
            raise LocalEntryNotFoundError("not cached")
        raise RequestsConnectionError("hub unreachable")

    load.calls = 0

    with pytest.raises(RequestsConnectionError):
        load_from_cache_or_hub(load, "thing")


def test_non_cache_error_is_not_retried():
    calls = []

    def load():
        calls.append(1)
        raise ValueError("bad argument")

    with pytest.raises(ValueError):
        load_from_cache_or_hub(load, "thing")
    assert len(calls) == 1
