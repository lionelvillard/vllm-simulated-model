import sys
from pathlib import Path
import pytest


# Add parent vllm venv to path if it exists
def _add_vllm_to_path():
    tests_dir = Path(__file__).parent
    repo_root = tests_dir.parent
    vllm_root = repo_root.parent / "vllm"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    vllm_site_packages = vllm_root / ".venv" / "lib" / f"python{python_version}" / "site-packages"

    if vllm_site_packages.exists() and str(vllm_site_packages) not in sys.path:
        sys.path.insert(0, str(vllm_site_packages))


_add_vllm_to_path()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
