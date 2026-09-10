import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repo root, regardless of where pytest was invoked from."""
    return Path(__file__).parent.resolve()


@pytest.fixture(scope="session")
def airpassengers_csv(repo_root: Path) -> str:
    """Absolute path to the local test dataset -- use this in tests instead of a hardcoded relative string."""
    return str(repo_root / "data" / "local" / "airpassengers.csv")