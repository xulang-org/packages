#!/usr/bin/env python3
"""Require an explicit, executable product compiler for package regression tests."""
import os
from pathlib import Path
import subprocess

def main():
    root = Path(os.environ.get("XU_ROOT", Path(__file__).resolve().parents[2] / "xu")).expanduser().resolve()
    compiler = Path(os.environ.get("XU", root / "bin/xu")).expanduser().resolve()
    if "bootstrap" in compiler.parts:
        raise SystemExit(f"package tests require the product compiler: {compiler}")
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise SystemExit(f"missing product compiler: {compiler}; set XU_ROOT or XU")
    result = subprocess.run([str(compiler), "--version"], capture_output=True, text=True, check=True)
    if "self-hosted" not in result.stdout:
        raise SystemExit(f"not a self-hosted product compiler: {compiler}")
    print(result.stdout.strip())

if __name__ == "__main__":
    main()
