#!/usr/bin/env python3
"""Rebuild the bundled native library; Python is a build tool only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PACKAGES = HERE.parents[1]
REPO = Path(os.environ.get("XU_ROOT", PACKAGES.parent / "xu")).expanduser().resolve()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", help="xunative target; defaults to both macOS architectures")
    parser.add_argument("--cc", default="auto")
    parser.add_argument("--xu-root", type=Path, default=REPO, help="compiler checkout containing scripts/xunative.py")
    args = parser.parse_args()
    source = json.loads((HERE.parent / "vendor/source.json").read_text())
    for name, expected in source["files"].items():
        actual = hashlib.sha256((HERE.parent / "vendor" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"vendored SQLite digest mismatch: {name}")
    native_tool = args.xu_root.expanduser().resolve() / "scripts/xunative.py"
    if not native_tool.is_file():
        raise SystemExit(f"missing {native_tool}; set XU_ROOT or --xu-root")
    tool = [sys.executable, str(native_tool)]
    with tempfile.TemporaryDirectory(prefix="xu-sqlite-native-") as temporary:
        slices = []
        for target in args.target or ["macos-arm64", "macos-x64"]:
            archive = Path(temporary) / f"sqlite-{target}.a"
            subprocess.run(tool + ["compile", "--manifest", str(HERE / "native.json"),
                "--target", target, "--cc", args.cc, "--cflag=-O2",
                "--cflag=-DSQLITE_THREADSAFE=1", "--cflag=-DSQLITE_OMIT_LOAD_EXTENSION=1",
                str(HERE / "bridge.c"), str(HERE.parent / "vendor/sqlite3.c"), "-o", str(archive)], check=True)
            slices += ["--slice", f"{target}={archive}"]
        subprocess.run(tool + ["pack", "--manifest", str(HERE / "native.json"), *slices, "-o", str(HERE / "sqlite.xulib")], check=True)

if __name__ == "__main__":
    main()
