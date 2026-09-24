"""Reuse the established data protocol and baseline modules."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "octa_baseline"))
spec = importlib.util.spec_from_file_location("ugr_paired_data", ROOT / "octa_oct_fusion/data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PairedDataset = module.PairedDataset

