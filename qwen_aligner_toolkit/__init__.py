from importlib.metadata import PackageNotFoundError, version

from .aligner import Aligner, Segment, Word
from .diarizer import Diarizer, SpeakerTurn
from .pipeline import Pipeline, PipelineResult
from .vad import VAD, merge_segments

try:
    __version__ = version("qwen-aligner-toolkit")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "Aligner",
    "Diarizer",
    "Pipeline",
    "PipelineResult",
    "Segment",
    "SpeakerTurn",
    "VAD",
    "Word",
    "merge_segments",
    "__version__",
]
