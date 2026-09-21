from pathlib import Path

import pytest


@pytest.fixture
def dataset_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "data" / "OCTA-500数据集"
    if not root.exists():
        pytest.skip(f"Dataset is not available at {root}")
    return root

