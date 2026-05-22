# qwen-aligner-toolkit

Production toolkit around **Qwen3-ForcedAligner**: VAD pre-segmentation,
multi-language word/char-level forced alignment, and speaker diarization.

## Why

[`qwen-asr`](https://pypi.org/project/qwen-asr/) ships an excellent forced
aligner (`Qwen3-ForcedAligner-0.6B`) that handles multiple languages with
one model — no per-language wav2vec2 needed, no kanji vocab gaps. This
toolkit packages it with the surrounding pieces you typically need in
production:

- **VAD** (`pyannote/segmentation-3.0`) to chunk long audio
- **Diarization** (`pyannote/speaker-diarization-community-1`) with
  per-word speaker assignment and run smoothing
- **Audio utilities** (path / URL / bytes / np.ndarray → 16 kHz mono)
- **AVX-less CPU compatibility** via a nagisa char-level stub

The toolkit does **not** do ASR. Bring your own transcript (from Whisper
via vLLM, faster-whisper, qwen-asr itself, or any other system) and the
toolkit will time-align it and label speakers.

## Install

```bash
pip install qwen-aligner-toolkit
pip install qwen-aligner-toolkit[full]    # with VAD + diarization
```

## Usage

### Simple alignment (short audio)

```python
from qwen_aligner_toolkit import Aligner

aligner = Aligner.from_pretrained()
words = aligner.align(
    text="甚至出现交易几乎停滞的情况。",
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    language="Chinese",
)
for w in words:
    print(w.start_time, w.end_time, w.text)
```

### ASR segments → word-level + speakers

```python
from qwen_aligner_toolkit import Pipeline

pipeline = Pipeline.from_pretrained(hf_token="hf_...", device="cuda")

segments = [
    {"text": "おはようございます。", "start": 0.0, "end": 2.3},
    {"text": "今日は良い天気ですね。", "start": 2.3, "end": 5.1},
]

result = pipeline.align_segments(
    segments=segments,
    audio="audio.wav",
    language="Japanese",
    diarize=True,
)

for w in result.words:
    print(f"{w.start_time:.2f}-{w.end_time:.2f} [{w.speaker}] {w.text}")
```

### VAD only

```python
from qwen_aligner_toolkit import VAD

vad = VAD.from_pretrained(hf_token="hf_...")
chunks = vad.detect("audio.wav")
```

### Sharing the segmentation model between VAD and Diarizer

`pyannote/speaker-diarization-community-1` already loads a copy of
`pyannote/segmentation-3.0` internally. To avoid loading it twice when you
also need standalone VAD, build the `VAD` on top of the diarizer's
segmentation submodel:

```python
from qwen_aligner_toolkit import Diarizer, VAD

diarizer = Diarizer.from_pretrained(hf_token="hf_...")
vad = VAD.from_segmentation_model(diarizer.segmentation_model)
chunks = vad.detect("audio.wav")
```

### Per-speaker grouped output

If you want speaker turns with joined text (one entry per consecutive
same-speaker run, with short-run smoothing), call
`Diarizer.split_words_by_speaker` on word-level dicts:

```python
from qwen_aligner_toolkit import Diarizer

words = [
    {"word": "hi",  "start": 0.0, "end": 1.0, "speaker": "A"},
    {"word": "yes", "start": 1.0, "end": 2.0, "speaker": "B"},
]
turns = Diarizer.split_words_by_speaker(words, min_duration_sec=0.3)
# [{"text": "hi", "start_time": 0.0, "end_time": 1.0, "speaker": "A"},
#  {"text": "yes", "start_time": 1.0, "end_time": 2.0, "speaker": "B"}]
```

For CJK languages where you don't want a space between concatenated tokens,
pass `join_separator=""`. The dict keys are configurable via `word_key` /
`start_key` / `end_key` / `speaker_key`.

## Configuration knobs

The toolkit has no global config — every knob is a function parameter. The
table below maps common production env-var conventions to the corresponding
toolkit argument, so you can wire them up with one-line plumbing.

### VAD

| Env var | Default | Toolkit argument |
|---------|---------|------------------|
| `ASR_VAD_ONSET` | `0.5` | `VAD.detect(onset=...)` |
| `ASR_VAD_OFFSET` | `0.5` | `VAD.detect(offset=...)` |
| `ASR_VAD_MIN_DURATION_ON` | `0.25` | `VAD.detect(min_duration_on=...)` |
| `ASR_VAD_MIN_DURATION_OFF` | `0.5` | `VAD.detect(min_duration_off=...)` |
| `ASR_VAD_MAX_CHUNK_SEC` | `30.0` | `merge_segments(max_chunk_sec=...)` / `Pipeline.vad_chunks(max_chunk_sec=...)` |
| `ASR_VAD_MAX_GAP_SEC` | `0.5` | `merge_segments(max_gap_sec=...)` |
| `ASR_VAD_PADDING_SEC` | `0.2` | `Aligner.align_segments(padding_sec=...)` |

### Diarization

| Env var | Default | Toolkit argument |
|---------|---------|------------------|
| `HF_TOKEN` | — | `Diarizer.from_pretrained(hf_token=...)` |
| `ASR_DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | `Diarizer.from_pretrained(model_id=...)` |
| `ASR_SPEAKER_MIN_DURATION_SEC` | `0.3` | `Diarizer.split_words_by_speaker(min_duration_sec=...)` |

### Device / sample rate

| Env var | Default | Toolkit argument |
|---------|---------|------------------|
| `ASR_DEVICE` | `cuda` (or `cpu` if no GPU) | `Aligner.from_pretrained(device_map=...)`, `Diarizer.from_pretrained(device=...)`, `VAD.from_pretrained(device=...)`, `Pipeline.from_pretrained(device=...)` |

All `device=` / `device_map=` arguments accept `None` to auto-detect
(`cuda` if available, else `cpu`).

The toolkit operates internally at **16 kHz mono**. Audio is auto-resampled
on load; there is no `target_sample_rate` knob exposed at the public API.

### Feature toggles

Flags like `ASR_VAD_ENABLED` or `ASR_DIARIZATION_ENABLED` belong to the
orchestration layer of your application, not the toolkit. The toolkit
exposes capabilities as separate classes (`VAD`, `Diarizer`) and the
`Pipeline` constructor takes explicit `with_vad=` / `with_diarization=`
flags:

```python
Pipeline.from_pretrained(
    hf_token=HF_TOKEN if ASR_DIARIZATION_ENABLED else None,
    with_vad=ASR_VAD_ENABLED,
    with_diarization=ASR_DIARIZATION_ENABLED,
)
```

## CPU compatibility (AVX requirement)

`qwen-asr` depends on `nagisa`, which ships `DyNet38` compiled with AVX.
On AVX-less CPUs (Intel Celeron G-series, some embedded SoCs), importing
`nagisa` crashes with `SIGILL`.

If that affects you, install the char-level stub:

```bash
qwen-aligner-toolkit install-nagisa-stub
# or
python -m qwen_aligner_toolkit.compat install-stub
```

This downgrades Japanese tokenization from morpheme-level to character-level
(other languages are unaffected). For forced alignment the difference is
mostly cosmetic; speaker boundaries may even be detected at finer
granularity.

For Docker:

```dockerfile
RUN pip install qwen-aligner-toolkit && \
    qwen-aligner-toolkit install-nagisa-stub
```

## License

Apache 2.0. See [LICENSE](LICENSE).
