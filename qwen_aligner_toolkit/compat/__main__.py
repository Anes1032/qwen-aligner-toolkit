import sys

from .nagisa_stub import install_stub


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m qwen_aligner_toolkit.compat install-stub")
        return 1
    if sys.argv[1] == "install-stub":
        path = install_stub()
        print(f"Installed nagisa stub at {path}")
        return 0
    print(f"Unknown subcommand: {sys.argv[1]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
