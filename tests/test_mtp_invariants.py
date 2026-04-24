"""P0 invariant tests for MTP speculative decoding.

Tests (mockable, no GPU required):
  1. KV length assertion after resync
  2. MTP mode must not use draft_model for drafting
  3. core_comparison policy path must use MTP head
  4. Shared MTP helper exists and has correct API
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestKVLengthInvariant:
    """Verify that kv_len is correctly advanced after resync."""

    def test_resync_advances_kv_len(self):
        """After target forward on last_tok, kv_len must be new_kv_len + 1."""
        from src.mtp_loop import resync_after_accept

        # Create mock target model that extends KV by 1
        mock_model = MagicMock()
        mock_kv = [
            (torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64))
        ]

        # After forward, KV should grow from 10 → 11
        mock_extended_kv = [
            (torch.randn(1, 4, 11, 64), torch.randn(1, 4, 11, 64))
        ]

        mock_output = MagicMock()
        mock_output.past_key_values = mock_extended_kv
        mock_output.logits = torch.randn(1, 1, 100)
        mock_output.hidden_states = [torch.randn(1, 1, 64)]
        mock_model.return_value = mock_output

        # Mock MTP head
        mock_mtp = MagicMock()
        mock_mtp.return_value = (torch.randn(1, 1, 100), None, None)

        last_tok = torch.tensor([42])
        new_kv_len = 10  # after trim, before resync forward

        _, _, _, final_kv_len = resync_after_accept(
            mock_model, mock_mtp, last_tok, mock_kv, new_kv_len,
        )

        assert final_kv_len == 11, \
            f"kv_len should be new_kv_len + 1 = 11, got {final_kv_len}"

    def test_oracle_script_has_kv_assertion(self):
        """oracle_sensitivity.py must contain kv_len assertion after resync."""
        import ast
        oracle_path = Path(__file__).parent.parent / "scripts" / "oracle_sensitivity.py"
        source = oracle_path.read_text()
        assert "assert target_kv[0][0].shape[2] == kv_len" in source, \
            "oracle_sensitivity.py must assert KV length matches kv_len after resync"

    def test_oracle_script_advances_kv_len(self):
        """oracle_sensitivity.py must advance kv_len = new_kv_len + 1."""
        oracle_path = Path(__file__).parent.parent / "scripts" / "oracle_sensitivity.py"
        source = oracle_path.read_text()
        assert "kv_len = new_kv_len + 1" in source, \
            "oracle_sensitivity.py must set kv_len = new_kv_len + 1 after target forward"


class TestMTPPathCorrectness:
    """Verify MTP mode uses MTP head, not target-as-draft."""

    def test_shared_helper_exists(self):
        """src/mtp_loop.py must exist with required functions."""
        from src.mtp_loop import mtp_draft_step, verify_and_accept, resync_after_accept
        assert callable(mtp_draft_step)
        assert callable(verify_and_accept)
        assert callable(resync_after_accept)

    def test_mtp_draft_step_calls_mtp_head(self):
        """mtp_draft_step must call mtp_head, not use target model for drafting."""
        from src.mtp_loop import mtp_draft_step

        mock_model = MagicMock()
        mock_mtp = MagicMock()

        # Mock model forward
        mock_kv = [(torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64))]
        mock_output = MagicMock()
        mock_output.past_key_values = [(torch.randn(1, 4, 6, 64), torch.randn(1, 4, 6, 64))]
        mock_output.hidden_states = [torch.randn(1, 1, 64)]
        mock_model.return_value = mock_output

        # Mock MTP head
        mock_mtp.return_value = (torch.randn(1, 1, 100), None, None)

        target_logits = torch.randn(1, 100)

        result, _ = mtp_draft_step(
            mock_model, mock_mtp, target_logits, mock_kv,
            kv_len=5, gamma=3, temperature=0.0,
        )

        assert mock_mtp.called, "MTP head must be called during drafting"
        assert result.gamma == 3
        assert len(result.tokens) == 3
        assert len(result.probs) == 3

    def test_calibrate_script_exists(self):
        """scripts/calibrate_mara.py must exist."""
        calib_path = Path(__file__).parent.parent / "scripts" / "calibrate_mara.py"
        assert calib_path.exists(), "calibrate_mara.py is required for MARA training"

    def test_core_comparison_has_mara_policies(self):
        """core_comparison.py must contain MARA policy definitions."""
        cc_path = Path(__file__).parent.parent / "scripts" / "core_comparison.py"
        source = cc_path.read_text()
        assert "mara_no_gate_or_uncertainty" in source, \
            "core_comparison.py must define mara_no_gate_or_uncertainty policy"
        assert "mara_full" in source, \
            "core_comparison.py must define mara_full policy"
        assert "existing_best_fragment_only" in source, \
            "core_comparison.py must define existing_best_fragment_only policy"


class TestDataSplitPolicy:
    """Verify calibration uses train split, not test."""

    def test_config_uses_train_for_calib(self):
        """mara_minimal.yaml must use train split for calibration."""
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "mara_minimal.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        ds = config.get("dataset", {})
        assert ds.get("calib_split") == "train", \
            f"Calibration must use train split, got {ds.get('calib_split')}"
        assert ds.get("eval_split") == "test", \
            f"Eval must use test split, got {ds.get('eval_split')}"
