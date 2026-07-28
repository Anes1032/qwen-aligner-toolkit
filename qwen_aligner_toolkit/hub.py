from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def forced_offline():
    """Force Hub lookups to resolve from the local cache for the duration of the block.

    Both flags are module-level snapshots taken at import time, so setting the
    HF_HUB_OFFLINE environment variable after import has no effect. Sessions are
    reset on the way in and out because the offline adapter is mounted once, when
    a cached session is first created.
    """
    import huggingface_hub.constants as hub_constants
    from huggingface_hub.utils._http import reset_sessions

    hub_previous = hub_constants.HF_HUB_OFFLINE
    hub_constants.HF_HUB_OFFLINE = True
    reset_sessions()

    try:
        import transformers.utils.hub as transformers_hub
    except ImportError:
        transformers_hub = None
        transformers_previous = None
    else:
        transformers_previous = transformers_hub._is_offline_mode
        transformers_hub._is_offline_mode = True

    try:
        yield
    finally:
        hub_constants.HF_HUB_OFFLINE = hub_previous
        reset_sessions()
        if transformers_hub is not None:
            transformers_hub._is_offline_mode = transformers_previous


def _is_cache_miss_or_unreachable(error: Exception) -> bool:
    from requests.exceptions import RequestException

    if isinstance(error, (RequestException, OSError)):
        return True
    try:
        from huggingface_hub import errors as hub_errors
    except ImportError:
        return False
    return isinstance(error, (hub_errors.HfHubHTTPError, hub_errors.LocalEntryNotFoundError))


def load_from_cache_or_hub(load: Callable[[], T], what: str) -> T:
    """Load a model from the local cache, falling back to the Hub on a cache miss.

    Going to the cache first keeps a fully cached model loadable when the Hub is
    unreachable. Some loaders query the Hub API unconditionally (transformers'
    ``fix_mistral_regex`` path calls ``model_info``), so a network-first attempt
    would exhaust its retry backoff before any fallback could run.

    The cached revision is used as-is; the Hub is only contacted when a required
    file is missing locally.
    """
    try:
        with forced_offline():
            return load()
    except Exception as error:
        if not _is_cache_miss_or_unreachable(error):
            raise
        logger.info(f"{what}: not in the local cache ({error}); fetching from the Hub")
        return load()
