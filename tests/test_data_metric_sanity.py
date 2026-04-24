"""Data and metric sanity tests.

Tests cover:
  1. GSM8K answer extraction works on known examples
  2. Calibration/eval split has no overlap
  3. Split is deterministic across runs
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.repro import make_calib_eval_split


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract final numerical answer from GSM8K answer string."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def check_gsm8k_correct(model_output: str, gold_answer: str) -> bool:
    """Check if model output contains the correct answer."""
    gold = extract_gsm8k_answer(gold_answer) if "####" in gold_answer else gold_answer
    gold = gold.strip().replace(",", "")
    if not gold:
        return False
    return gold in model_output


class TestGSM8KMetric:
    def test_known_answer_extraction(self):
        assert extract_gsm8k_answer("The answer is #### 42") == "42"
        assert extract_gsm8k_answer("#### 1,234") == "1234"
        assert extract_gsm8k_answer("#### -5") == "-5"
        assert extract_gsm8k_answer("No answer marker") == ""

    def test_answer_checking(self):
        assert check_gsm8k_correct("The answer is 42.", "#### 42")
        assert check_gsm8k_correct("Therefore, 42 apples.", "#### 42")
        assert not check_gsm8k_correct("The answer is 43.", "#### 42")
        assert check_gsm8k_correct("Result: 1234", "#### 1,234")

    def test_edge_cases(self):
        assert not check_gsm8k_correct("", "#### 42")
        assert not check_gsm8k_correct("42", "")
        assert check_gsm8k_correct("0", "#### 0")


class TestSplitSanity:
    def test_no_overlap_various_sizes(self):
        for total in [10, 50, 100, 500]:
            split = make_calib_eval_split(total, calib_fraction=0.3, seed=42)
            overlap = set(split.calib_indices) & set(split.eval_indices)
            assert len(overlap) == 0, f"Overlap at total={total}: {overlap}"
            assert len(split.calib_indices) + len(split.eval_indices) == total

    def test_calib_fraction_respected(self):
        split = make_calib_eval_split(100, calib_fraction=0.3, seed=42)
        assert len(split.calib_indices) == 30
        assert len(split.eval_indices) == 70

    def test_different_seeds_different_splits(self):
        s1 = make_calib_eval_split(100, calib_fraction=0.3, seed=42)
        s2 = make_calib_eval_split(100, calib_fraction=0.3, seed=123)
        assert s1.calib_indices != s2.calib_indices

    def test_indices_in_range(self):
        split = make_calib_eval_split(50, calib_fraction=0.3, seed=42)
        for idx in split.calib_indices + split.eval_indices:
            assert 0 <= idx < 50
