#!/usr/bin/env python3
"""Require an explicit, executable product compiler for package regression tests."""
import os
import hashlib
import importlib.util
from pathlib import Path
import subprocess

def main():
    root = Path(os.environ.get("XU_ROOT", Path(__file__).resolve().parents[2] / "xu")).expanduser().resolve()
    compiler = Path(os.environ.get("XU", root / "bin/xu")).expanduser().resolve()
    if "bootstrap" in compiler.parts:
        raise SystemExit(f"package tests require the product compiler: {compiler}")
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise SystemExit(f"missing product compiler: {compiler}; set XU_ROOT or XU")
    # A release banner is presentation, not compiler provenance. Use the
    # selected checkout's canonical product resolver and provenance policy.
    resolver = root / 'scripts/xu_host_seed.sh'
    if not resolver.is_file():
        raise SystemExit(f"missing product compiler verifier: {resolver}")
    verified = subprocess.run([str(resolver)], cwd=root,
        env=dict(os.environ, XU_SEED=str(compiler)), capture_output=True, text=True)
    if verified.returncode != 0:
        raise SystemExit(verified.stderr.strip() or 'product compiler verification failed')
    if Path(verified.stdout.strip()).resolve() != compiler.resolve():
        raise SystemExit('product compiler verifier selected a different executable')
    if Path(str(compiler) + '.provenance').is_file():
        # An installed SDK may lack a sidecar; when present it must describe
        # this binary's successful fixed point, not just a product-lane banner.
        spec = importlib.util.spec_from_file_location('xu_provenance', root / 'scripts/verify_product_provenance.py')
        provenance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(provenance)
        try:
            fields = provenance.read_provenance(compiler)
            if fields.get('status') != 'PASS' or fields.get('lane') != 'product':
                raise ValueError('provenance must identify a passing product build')
            provenance.fixed_point_identity(fields, hashlib.sha256(compiler.read_bytes()).hexdigest())
        except (ValueError, OSError) as error:
            raise SystemExit(f'product compiler verification failed: {error}')
    result = subprocess.run([str(compiler), "--version"], capture_output=True, text=True, check=True)
    print(result.stdout.strip())

if __name__ == "__main__":
    main()
