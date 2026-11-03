#!/usr/bin/env python3
"""Build optional Metal slices plus SDK-independent unavailable stubs."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DEFAULTS = ['macos-arm64','macos-x64','linux-arm64','linux-x64','windows-arm64','windows-x64']
TRIPLES = {'macos-arm64':'arm64-apple-macos11','macos-x64':'x86_64-apple-macos11',
           'linux-arm64':'aarch64-linux-gnu','linux-x64':'x86_64-linux-gnu',
           'windows-arm64':'aarch64-pc-windows-msvc','windows-x64':'x86_64-pc-windows-msvc',
           'linux-riscv64':'riscv64-linux-gnu','linux-loongarch64':'loongarch64-linux-gnu'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',action='append',choices=TRIPLES)
    parser.add_argument('--cc',help='clang executable; defaults to selected Xcode clang on macOS')
    parser.add_argument('--sdk',type=Path,help='matching macOS SDK for Metal slices')
    parser.add_argument('--xu-root',type=Path,default=Path(os.environ.get('XU_ROOT',HERE.parents[2]/'xu')))
    parser.add_argument('-o','--output',type=Path,default=HERE/'metal.xulib')
    args = parser.parse_args()
    root = args.xu_root.expanduser().resolve()
    cc = args.cc
    if not cc:
        cc = subprocess.check_output(['xcrun','--find','clang'],text=True).strip() if sys.platform=='darwin' else shutil.which('clang')
    if not cc: raise SystemExit('clang is required')
    targets = args.target or DEFAULTS
    sdk = args.sdk
    if any(t.startswith('macos-') for t in targets) and sdk is None:
        compiler = Path(cc).resolve()
        developer = next((p for p in compiler.parents if p.name=='Developer'),None)
        candidate = developer/'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk' if developer else None
        sdk = candidate if candidate and candidate.is_dir() else Path(subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip())
    tool = [sys.executable,str(root/'scripts/xunative.py')]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='xu-metal-native-') as temporary:
        work = Path(temporary);slices=[]
        for target in targets:
            obj=work/(target+'.o');archive=work/(target+'.a')
            command=[cc,'--target='+TRIPLES[target],'-O2','-I'+str(root/'include'),'-I'+str(HERE),'-c']
            if target.startswith('macos-'):
                if not sdk or not sdk.is_dir(): raise SystemExit('matching macOS SDK is required')
                command+=['-fobjc-arc','-fmodules','-fmodules-cache-path='+str(work/'modules'),'-isysroot',str(sdk),str(HERE/'bridge.m')]
            else: command+=[str(HERE/'unavailable.c')]
            subprocess.run(command+['-o',str(obj)],check=True)
            subprocess.run(tool+['archive','--cc',cc,'--manifest',str(HERE/'native.json'),'--target',target,'--native',str(obj),'-o',str(archive)],check=True)
            slices+=['--slice',target+'='+str(archive)]
        out=work/'metal.xulib'
        subprocess.run(tool+['pack','--manifest',str(HERE/'native.json'),*slices,'-o',str(out)],check=True)
        # Publish only after every requested slice has built and validated.
        staged=args.output.with_name(args.output.name+'.tmp')
        shutil.copyfile(out,staged);staged.replace(args.output)


if __name__=='__main__':main()
