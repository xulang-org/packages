"""Check independent package versions against Git or the local migration baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIELDS = re.compile(r'^(version|xu):[^\n]*(?:\n|$)', re.M)
SEMVER = r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)'

def version(text, field='version'):
    lines = re.findall(rf'^{field}:\s*!ver\s+([^\n]+)$', text, re.M)
    if len(lines) != 1 or not re.fullmatch(SEMVER, lines[0]):
        raise ValueError(f'expected one valid {field}: !ver x.y.z field')
    return tuple(map(int, lines[0].split('.')))

def digest(files):
    sha = hashlib.sha256()
    for name, content in sorted(files.items()):
        if name == 'xu.xun':
            content = FIELDS.sub('', content.decode()).encode()
        sha.update(name.encode() + b'\0' + hashlib.sha256(content).digest())
    return sha.hexdigest()

def inventories(names, read):
    manifests = {name for name in names if name == 'xu.xun' or name.endswith('/xu.xun')}
    result = {}
    for manifest in sorted(manifests):
        folder = str(Path(manifest).parent)
        prefix = '' if folder == '.' else folder+'/'
        files = {}
        for name in names:
            if not name or not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            # Nested modules own their own files, when present.
            if any(other != manifest and name.startswith(str(Path(other).parent)+'/')
                   for other in manifests):
                continue
            body = read(name)
            if body is not None:
                files[relative] = body
        body = read(manifest)
        if body is None:
            continue
        result[folder] = {'version': list(version(body.decode())),
                          'xu': list(version(body.decode(), 'xu')), 'digest': digest(files)}
    return result

def current(root):
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).decode().split('\0')
    return inventories(names, lambda name: (root/name).read_bytes() if (root/name).is_file() else None)

def baseline(root, base):
    exists = subprocess.run(['git', '-C', str(root), 'rev-parse', '--verify', base], capture_output=True).returncode == 0
    if not exists:
        if base != 'HEAD':
            raise ValueError(f'missing base revision: {base}')
        git_dir = Path(subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--absolute-git-dir']).decode().strip())
        path = git_dir / 'xu-module-baseline.json'
        if not path.is_file():
            raise ValueError('initial migration baseline missing; do not reset imported module versions')
        return json.loads(path.read_text())
    names = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '--name-only', '-z', base]).decode().split('\0')
    return inventories(names, lambda name: subprocess.check_output(['git', '-C', str(root), 'show', f'{base}:{name}']))

def process(root, base='HEAD', bump=False, kind='patch', release=None):
    old = baseline(root, base)
    modules = current(root)
    errors, updates = [], {}
    if not modules:
        return ['no package module manifests found']
    for name, info in sorted(modules.items()):
        value = tuple(info['version'])
        previous = tuple(old[name]['version']) if name in old else None
        target = value
        if previous is None and value != (0, 1, 0):
            errors.append(f'{name}: new module must start at 0.1.0')
        elif previous is not None and value < previous:
            errors.append(f'{name}: version regressed below {previous}')
        elif previous is not None and info['digest'] != old[name]['digest'] and value <= previous:
            if bump:
                a, b, c = previous
                target = {'patch': (a,b,c+1), 'minor': (a,b+1,0), 'major': (a+1,0,0)}[kind]
            else:
                errors.append(f'{name}: module changed but version did not increase past {".".join(map(str,previous))}')
        path = root / name / 'xu.xun'
        text = path.read_text()
        if target != value:
            text = re.sub(r'^version:[^\n]*', 'version: !ver '+'.'.join(map(str,target)), text, flags=re.M)
        if release is not None:
            if tuple(info['xu']) > release:
                errors.append(f'{name}: cannot regress xu:')
            text = re.sub(r'^xu:[^\n]*', 'xu: !ver '+'.'.join(map(str,release)), text, flags=re.M)
        if text != path.read_text():
            updates[path] = text
    if not errors:
        for path, text in updates.items():
            path.write_text(text)
            print(f'{path.parent.name}: version {".".join(map(str,version(text)))} / xu {".".join(map(str,version(text,"xu")))}')
    return errors

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='HEAD')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--bump', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--sync-release', action='store_true')
    parser.add_argument('--kind', choices=['patch','minor','major'], default='patch')
    args = parser.parse_args()
    try:
        release = None
        if args.sync_release:
            compiler = Path(os.environ.get('XU_ROOT', ROOT.parent/'xu'))
            release = version('version: !ver '+(compiler/'VERSION').read_text().strip()+'\n')
        errors = process(ROOT, args.base, args.bump, args.kind, release)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Package module versions OK')

if __name__ == '__main__':
    main()
