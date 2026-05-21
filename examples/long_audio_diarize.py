import os

from qwen_aligner_toolkit import Pipeline


def main() -> None:
    hf_token = os.environ["HF_TOKEN"]
    pipeline = Pipeline.from_pretrained(hf_token=hf_token, device="cuda")

    segments = [
        {"text": "おはようございます。", "start": 0.0, "end": 2.3},
        {"text": "今日は良い天気ですね。", "start": 2.3, "end": 5.1},
    ]

    result = pipeline.align_segments(
        segments=segments,
        audio="path/to/long_audio.wav",
        language="Japanese",
        diarize=True,
    )

    for w in result.words:
        speaker = w.speaker or "-"
        print(f"{w.start_time:6.2f} - {w.end_time:6.2f}  [{speaker:>10s}]  {w.text}")


if __name__ == "__main__":
    main()
