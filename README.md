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
    audio="audio.wav",
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
