"""Imported module versions survive migration and bumps remain idempotent."""
import importlib.util
import json
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/module_versions.py"
spec = importlib.util.spec_from_file_location("package_versions_test", SCRIPT)
versions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(versions)

def imported(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    module = tmp_path / "sqlite"
    module.mkdir()
    (module / "xu.xun").write_text("module: sqlite\nversion: !ver 3.0.0\nxu: !ver 0.8.1\n")
    (module / "mod.xu").write_text("original")
    (tmp_path / ".git/xu-module-baseline.json").write_text(json.dumps(versions.current(tmp_path)))
    return module

def test_imported_versions_and_xu_sync_do_not_bump_module(tmp_path):
    module = imported(tmp_path)
    assert versions.process(tmp_path) == []
    assert versions.process(tmp_path, release=(0,8,2)) == []
    assert versions.version((module / "xu.xun").read_text()) == (3,0,0)
    assert versions.process(tmp_path) == []

def test_changed_module_requires_one_bump(tmp_path):
    module = imported(tmp_path)
    (module / "mod.xu").write_text("changed")
    assert "did not increase" in versions.process(tmp_path)[0]
    assert versions.process(tmp_path, bump=True) == []
    assert versions.process(tmp_path, bump=True) == []
    assert versions.version((module / "xu.xun").read_text()) == (3,0,1)

def test_invalid_release_never_writes_partial_manifest(tmp_path):
    module = imported(tmp_path)
    original = (module / "xu.xun").read_bytes()
    assert versions.process(tmp_path, release=(0,7,0))
    assert (module / "xu.xun").read_bytes() == original


def test_root_module_owns_internal_packages_without_child_manifests(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "xu.xun").write_text('module: "github.com/xulang-org/packages"\nversion: !ver 0.1.0\nxu: !ver 0.8.2\n')
    (tmp_path / "mysql").mkdir()
    source = tmp_path / "mysql/mod.xu"
    source.write_text("original")
    (tmp_path / ".git/xu-module-baseline.json").write_text(json.dumps(versions.current(tmp_path)))
    assert list(versions.current(tmp_path)) == ['.']
    source.write_text("changed")
    assert versions.process(tmp_path)
    assert versions.process(tmp_path, bump=True) == []
    assert versions.version((tmp_path / "xu.xun").read_text()) == (0,1,1)
    assert versions.process(tmp_path, bump=True) == []


def test_official_package_identity_and_single_module_root():
    root = SCRIPT.parents[1]
    assert (root / "xu.xun").read_text().startswith('module: "github.com/xulang-org/packages"\n')
    assert not list(root.glob('*/xu.xun'))
    packages = [p for p in root.iterdir() if (p / 'mod.xu').is_file()]
    assert len(packages) == 29
    for package in packages:
        for source in package.glob('*.xu'):
            assert source.read_text().startswith('@"github.com/xulang-org/packages/'+package.name+'"')
