def test_public_api():
    from qwen_aligner_toolkit import (
        VAD,
        Aligner,
        Diarizer,
        Pipeline,
        PipelineResult,
        Segment,
        SpeakerTurn,
        Word,
        merge_segments,
    )
    assert all(
        [Aligner, Diarizer, Pipeline, PipelineResult, Segment, SpeakerTurn, VAD, Word, merge_segments]
    )


def test_version():
    import qwen_aligner_toolkit

    assert qwen_aligner_toolkit.__version__
