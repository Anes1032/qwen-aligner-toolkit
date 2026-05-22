from qwen_aligner_toolkit import Diarizer, SpeakerTurn


def _turns(*triples):
    return [SpeakerTurn(start=s, end=e, speaker=sp) for s, e, sp in triples]


def test_dominant_speaker_overlap():
    turns = _turns((0.0, 5.0, "A"), (3.0, 8.0, "B"))
    assert Diarizer.dominant_speaker(0.0, 4.0, turns) == "A"
    assert Diarizer.dominant_speaker(4.0, 8.0, turns) == "B"


def test_dominant_speaker_no_overlap_returns_none():
    turns = _turns((0.0, 1.0, "A"))
    assert Diarizer.dominant_speaker(5.0, 6.0, turns) is None


def test_dominant_speaker_fallback_nearest():
    turns = _turns((0.0, 1.0, "A"), (10.0, 11.0, "B"))
    assert Diarizer.dominant_speaker(5.0, 6.0, turns, fallback_nearest=True) == "A"
    assert Diarizer.dominant_speaker(8.0, 9.0, turns, fallback_nearest=True) == "B"


def test_dominant_speaker_empty_turns():
    assert Diarizer.dominant_speaker(0.0, 1.0, [], fallback_nearest=True) is None


def test_split_words_by_speaker_single_speaker():
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "A"},
        {"word": "world", "start": 0.5, "end": 1.0, "speaker": "A"},
    ]
    out = Diarizer.split_words_by_speaker(words, min_duration_sec=0.0)
    assert out == [
        {"text": "hello world", "start_time": 0.0, "end_time": 1.0, "speaker": "A"},
    ]


def test_split_words_by_speaker_two_speakers():
    words = [
        {"word": "hi", "start": 0.0, "end": 1.0, "speaker": "A"},
        {"word": "yes", "start": 1.0, "end": 2.0, "speaker": "B"},
    ]
    out = Diarizer.split_words_by_speaker(words, min_duration_sec=0.0)
    assert len(out) == 2
    assert out[0]["speaker"] == "A" and out[0]["text"] == "hi"
    assert out[1]["speaker"] == "B" and out[1]["text"] == "yes"


def test_split_words_by_speaker_cjk_no_separator():
    words = [
        {"word": "甚", "start": 0.0, "end": 0.3, "speaker": "A"},
        {"word": "至", "start": 0.3, "end": 0.6, "speaker": "A"},
    ]
    out = Diarizer.split_words_by_speaker(
        words, min_duration_sec=0.0, join_separator=""
    )
    assert out[0]["text"] == "甚至"


def test_split_words_by_speaker_absorbs_none_speaker():
    words = [
        {"word": "a", "start": 0.0, "end": 1.0, "speaker": "A"},
        {"word": "b", "start": 1.0, "end": 1.2, "speaker": None},
        {"word": "c", "start": 1.2, "end": 2.0, "speaker": "A"},
    ]
    out = Diarizer.split_words_by_speaker(words, min_duration_sec=0.0)
    assert len(out) == 1
    assert out[0]["speaker"] == "A"
    assert "a" in out[0]["text"] and "b" in out[0]["text"] and "c" in out[0]["text"]


def test_split_words_by_speaker_empty_input():
    assert Diarizer.split_words_by_speaker([], min_duration_sec=0.3) == []


def test_split_words_by_speaker_custom_keys():
    words = [
        {"text": "hello", "start_time": 0.0, "end_time": 1.0, "spk": "A"},
    ]
    out = Diarizer.split_words_by_speaker(
        words,
        min_duration_sec=0.0,
        word_key="text",
        start_key="start_time",
        end_key="end_time",
        speaker_key="spk",
    )
    assert out == [
        {"text": "hello", "start_time": 0.0, "end_time": 1.0, "speaker": "A"},
    ]


def test_split_words_by_speaker_drops_empty_text():
    words = [
        {"word": "", "start": 0.0, "end": 1.0, "speaker": "A"},
    ]
    assert Diarizer.split_words_by_speaker(words, min_duration_sec=0.0) == []
