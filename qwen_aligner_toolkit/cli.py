from __future__ import annotations

import argparse
import sys

from .compat.nagisa_stub import install_stub


def main() -> int:
    parser = argparse.ArgumentParser(prog="qwen-aligner-toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("install-nagisa-stub", help="Replace nagisa with a char-level stub (for AVX-less CPUs).")

    args = parser.parse_args()
    if args.command == "install-nagisa-stub":
        path = install_stub()
        print(f"Installed nagisa stub at {path}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
