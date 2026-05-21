from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys

STUB_SOURCE = '''class _Tagged:
    def __init__(self, words):
        self.words = words


def tagging(text):
    return _Tagged(list(text))
'''


def install_stub() -> str:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "nagisa", "DyNet38"],
        check=False,
    )
    target_root = site.getsitepackages()[0]
    target_dir = os.path.join(target_root, "nagisa")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "__init__.py")
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(STUB_SOURCE)
    return target_file
