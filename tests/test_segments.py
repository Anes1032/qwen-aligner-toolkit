from qwen_aligner_toolkit import merge_segments


def test_merge_empty():
    assert merge_segments([]) == []


def test_merge_close_segments():
    segs = [{"start": 0.0, "end": 2.0}, {"start": 2.2, "end": 5.0}]
    merged = merge_segments(segs, max_chunk_sec=30, max_gap_sec=0.5)
    assert merged == [{"start": 0.0, "end": 5.0}]


def test_keep_segments_with_large_gap():
    segs = [{"start": 0.0, "end": 1.0}, {"start": 5.0, "end": 6.0}]
    merged = merge_segments(segs, max_chunk_sec=30, max_gap_sec=0.5)
    assert len(merged) == 2
    assert merged[0] == {"start": 0.0, "end": 1.0}
    assert merged[1] == {"start": 5.0, "end": 6.0}


def test_split_long_segment():
    segs = [{"start": 0.0, "end": 95.0}]
    out = merge_segments(segs, max_chunk_sec=30, max_gap_sec=0.5)
    assert len(out) == 4
    assert out[0]["start"] == 0.0
    assert out[-1]["end"] == 95.0
    for chunk in out:
        assert chunk["end"] - chunk["start"] <= 30.0 + 1e-6


def test_merge_then_split_combination():
    segs = [
        {"start": 0.0, "end": 10.0},
        {"start": 10.3, "end": 20.0},
        {"start": 20.2, "end": 50.0},
    ]
    out = merge_segments(segs, max_chunk_sec=30, max_gap_sec=0.5)
    assert out[0]["start"] == 0.0
    assert out[-1]["end"] == 50.0
    for chunk in out:
        assert chunk["end"] - chunk["start"] <= 30.0 + 1e-6
