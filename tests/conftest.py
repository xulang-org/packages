import importlib.util
from pathlib import Path

def pytest_sessionstart(session):
    path = Path(__file__).resolve().parents[1] / "scripts/check_product_compiler.py"
    spec = importlib.util.spec_from_file_location("package_product_compiler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
