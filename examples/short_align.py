from qwen_aligner_toolkit import Aligner


def main() -> None:
    aligner = Aligner.from_pretrained()
    words = aligner.align(
        text="甚至出现交易几乎停滞的情况。",
        audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
        language="Chinese",
    )
    for w in words:
        print(f"{w.start_time:6.2f} - {w.end_time:6.2f}  {w.text}")


if __name__ == "__main__":
    main()
