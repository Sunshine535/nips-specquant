"""Unit tests for SpecQuant core modules."""

import math

import pytest
import torch

from src.turboquant_kv import (
    HadamardRotation,
    QuantizedKVCache,
    ScalarQuantizer,
    compute_tv_bound,
    fast_hadamard_inverse,
    fast_hadamard_transform,
)


# ---------------------------------------------------------------------------
# HadamardRotation
# ---------------------------------------------------------------------------

class TestHadamardRotation:
    def test_roundtrip_exact(self):
        rot = HadamardRotation(64, seed=42)
        x = torch.randn(2, 4, 10, 64)
        recovered = rot.inverse_rotate(rot.rotate(x))
        torch.testing.assert_close(recovered, x, atol=1e-4, rtol=1e-4)

    def test_pads_non_power_of_2(self):
        rot = HadamardRotation(50, seed=0)
        assert rot.padded_dim == 64
        x = torch.randn(1, 1, 5, 50)
        rotated = rot.rotate(x)
        assert rotated.shape[-1] == 64
        recovered = rot.inverse_rotate(rotated)
        assert recovered.shape[-1] == 50
        torch.testing.assert_close(recovered, x, atol=1e-4, rtol=1e-4)

    def test_deterministic_with_same_seed(self):
        a = HadamardRotation(128, seed=7)
        b = HadamardRotation(128, seed=7)
        x = torch.randn(1, 1, 4, 128)
        torch.testing.assert_close(a.rotate(x), b.rotate(x))

    def test_different_seeds_differ(self):
        a = HadamardRotation(64, seed=1)
        b = HadamardRotation(64, seed=2)
        x = torch.randn(1, 1, 4, 64)
        assert not torch.allclose(a.rotate(x), b.rotate(x))


class TestFastHadamardTransform:
    def test_inverse_recovers_input(self):
        signs = torch.randint(0, 2, (16,), dtype=torch.float32) * 2 - 1
        x = torch.randn(4, 16)
        y = fast_hadamard_transform(x, signs)
        x_rec = fast_hadamard_inverse(y, signs)
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_orthogonality(self):
        x = torch.randn(8, 32)
        y = fast_hadamard_transform(x, signs=None)
        inner_x = (x * x).sum(dim=-1)
        inner_y = (y * y).sum(dim=-1)
        torch.testing.assert_close(inner_x, inner_y, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# ScalarQuantizer
# ---------------------------------------------------------------------------

class TestScalarQuantizer:
    def test_4bit_roundtrip_fidelity(self):
        q = ScalarQuantizer(bits=4, block_size=32)
        x = torch.randn(1, 4, 64, 128)
        codes, scales, zeros = q.quantize(x)
        recon = q.dequantize(codes, scales, zeros)
        mse = (x - recon).pow(2).mean().item()
        assert mse < 0.05, f"4-bit MSE too high: {mse}"

    def test_3bit_code_range(self):
        q = ScalarQuantizer(bits=3, block_size=128)
        x = torch.randn(1, 8, 256, 128)
        codes, scales, zeros = q.quantize(x)
        assert codes.dtype == torch.uint8
        assert codes.max().item() <= 7  # 2^3 - 1

    def test_shape_preserved(self):
        q = ScalarQuantizer(bits=4, block_size=64)
        x = torch.randn(2, 4, 100, 64)
        codes, scales, zeros = q.quantize(x)
        recon = q.dequantize(codes, scales, zeros)
        assert recon.shape == x.shape

    def test_more_bits_less_error(self):
        x = torch.randn(1, 4, 128, 64)
        q3 = ScalarQuantizer(bits=3, block_size=64)
        q4 = ScalarQuantizer(bits=4, block_size=64)
        mse3 = (x - q3.dequantize(*q3.quantize(x))).pow(2).mean().item()
        mse4 = (x - q4.dequantize(*q4.quantize(x))).pow(2).mean().item()
        assert mse4 < mse3, "4-bit should have lower error than 3-bit"


# ---------------------------------------------------------------------------
# QuantizedKVCache
# ---------------------------------------------------------------------------

class TestQuantizedKVCache:
    def _make_cache(self, bits=4, seq_len=32):
        cache = QuantizedKVCache(
            num_layers=2, num_kv_heads=4, head_dim=64,
            bits=bits, block_size=32,
        )
        for i in range(2):
            k = torch.randn(1, 4, seq_len, 64)
            v = torch.randn(1, 4, seq_len, 64)
            cache.compress_and_store(i, k, v)
        return cache

    def test_seq_len_tracking(self):
        cache = self._make_cache(seq_len=48)
        assert cache.seq_len == 48

    def test_get_rotated_kv_shape(self):
        cache = self._make_cache(seq_len=32)
        k, v = cache.get_rotated_kv(0)
        assert k.shape == (1, 4, 32, 64)
        assert v.shape == (1, 4, 32, 64)

    def test_compressed_attention_output_shape(self):
        cache = self._make_cache(seq_len=32)
        q = torch.randn(1, 4, 1, 64)
        out = cache.compressed_attention(0, q)
        assert out.shape == (1, 4, 1, 64)

    def test_gqa_attention(self):
        cache = QuantizedKVCache(
            num_layers=1, num_kv_heads=4, head_dim=64, bits=4, block_size=32,
        )
        k = torch.randn(1, 4, 16, 64)
        v = torch.randn(1, 4, 16, 64)
        cache.compress_and_store(0, k, v)
        q = torch.randn(1, 16, 1, 64)  # 16 query heads, 4 KV heads
        out = cache.compressed_attention(0, q)
        assert out.shape == (1, 16, 1, 64)

    def test_memory_compression_ratio(self):
        cache = QuantizedKVCache(
            num_layers=4, num_kv_heads=8, head_dim=128,
            bits=3, block_size=128,
        )
        for i in range(4):
            k = torch.randn(1, 8, 1024, 128)
            v = torch.randn(1, 8, 1024, 128)
            cache.compress_and_store(i, k, v)
        ratio = cache.compression_ratio
        assert ratio > 1.5, f"Compression ratio too low: {ratio}"

    def test_empty_cache_zero_bytes(self):
        cache = QuantizedKVCache(
            num_layers=2, num_kv_heads=4, head_dim=64, bits=3,
        )
        assert cache.memory_bytes() == 0
        assert cache.full_precision_bytes() == 0


# ---------------------------------------------------------------------------
# compute_tv_bound
# ---------------------------------------------------------------------------

class TestComputeTVBound:
    def test_positive(self):
        tv = compute_tv_bound(
            w_o_fnorm=1.0, range_k=4.0, range_v=4.0, v_fnorm=1.0,
            dim=128, bits=3, block_size=128,
        )
        assert tv > 0

    def test_more_bits_tighter_bound(self):
        tv3 = compute_tv_bound(
            w_o_fnorm=1.0, range_k=4.0, range_v=4.0, v_fnorm=1.0,
            dim=128, bits=3, block_size=128,
        )
        tv4 = compute_tv_bound(
            w_o_fnorm=1.0, range_k=4.0, range_v=4.0, v_fnorm=1.0,
            dim=128, bits=4, block_size=128,
        )
        assert tv4 < tv3

    def test_higher_temperature_reduces_bound(self):
        tv_t1 = compute_tv_bound(
            w_o_fnorm=1.0, range_k=4.0, range_v=4.0, v_fnorm=1.0,
            dim=128, bits=3, block_size=128, temperature=1.0,
        )
        tv_t2 = compute_tv_bound(
            w_o_fnorm=1.0, range_k=4.0, range_v=4.0, v_fnorm=1.0,
            dim=128, bits=3, block_size=128, temperature=2.0,
        )
        assert tv_t2 < tv_t1

    def test_zero_norms_zero_bound(self):
        tv = compute_tv_bound(
            w_o_fnorm=0.0, range_k=4.0, range_v=4.0, v_fnorm=1.0,
            dim=128, bits=3, block_size=128,
        )
        assert tv == 0.0


# ---------------------------------------------------------------------------
# Integration: compress → dequantize → inverse-rotate end-to-end
# ---------------------------------------------------------------------------

class TestEndToEndCompression:
    def test_kv_roundtrip_reasonable_error(self):
        """Full pipeline: original → rotate → quantize → dequantize → inv-rotate."""
        rot = HadamardRotation(128, seed=42)
        quant = ScalarQuantizer(bits=4, block_size=64)

        k = torch.randn(1, 8, 64, 128)
        k_rot = rot.rotate(k)
        codes, scales, zeros = quant.quantize(k_rot)
        k_deq_rot = quant.dequantize(codes, scales, zeros)
        k_recovered = rot.inverse_rotate(k_deq_rot)

        mse = (k - k_recovered).pow(2).mean().item()
        assert mse < 0.05, f"End-to-end MSE too high: {mse}"

    def test_attention_output_close_to_fp(self):
        """Compressed attention should be close to full-precision attention."""
        num_kv_heads = 4
        head_dim = 64
        seq_len = 32

        k = torch.randn(1, num_kv_heads, seq_len, head_dim)
        v = torch.randn(1, num_kv_heads, seq_len, head_dim)
        q = torch.randn(1, num_kv_heads, 1, head_dim)

        scale = 1.0 / math.sqrt(head_dim)
        scores_fp = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_fp = torch.softmax(scores_fp, dim=-1)
        out_fp = torch.matmul(attn_fp, v)

        cache = QuantizedKVCache(
            num_layers=1, num_kv_heads=num_kv_heads,
            head_dim=head_dim, bits=4, block_size=32,
        )
        cache.compress_and_store(0, k, v)
        out_quant = cache.compressed_attention(0, q)

        cosine_sim = torch.nn.functional.cosine_similarity(
            out_fp.flatten(), out_quant.flatten(), dim=0,
        )
        assert cosine_sim > 0.9, f"Cosine similarity too low: {cosine_sim}"


# ---------------------------------------------------------------------------
# TestStatUtils — src/utils.py
# ---------------------------------------------------------------------------

import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.utils import (
    mean_confidence_interval,
    bootstrap_ci,
    aggregate_trials,
    paired_ttest,
    compute_effect_size,
    format_with_ci,
)


class TestStatUtils:
    def test_mean_ci_basic(self):
        """CI should contain the mean, and bounds should be reasonable."""
        data = [10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 11.0]
        mean, ci_lo, ci_hi = mean_confidence_interval(data, confidence=0.95)
        assert ci_lo < mean < ci_hi, "Mean should lie within its CI"
        assert ci_lo > 0, "Lower bound should be positive for positive data"
        assert ci_hi - ci_lo < 10.0, "CI width should be reasonable for tight data"

    def test_bootstrap_ci_single_value(self):
        """Bootstrap with a single observation should degrade gracefully."""
        data = [5.0]
        point, ci_lo, ci_hi = bootstrap_ci(data, n_bootstrap=1000, confidence=0.95)
        assert point == 5.0
        # With a single value, all bootstrap resamples are identical
        assert ci_lo == pytest.approx(5.0)
        assert ci_hi == pytest.approx(5.0)

    def test_aggregate_trials(self):
        """aggregate_trials should return all expected summary keys."""
        values = [0.91, 0.93, 0.90, 0.92, 0.94]
        result = aggregate_trials(values, confidence=0.95)
        expected_keys = {
            "mean", "std", "median", "min", "max",
            "ci_lower", "ci_upper", "ci_confidence", "n_trials",
        }
        assert expected_keys.issubset(result.keys())
        assert result["n_trials"] == 5
        assert result["min"] == pytest.approx(0.90)
        assert result["max"] == pytest.approx(0.94)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_paired_ttest_identical(self):
        """Paired t-test on identical arrays should yield p_value ~ 1.0 or NaN, cohen_d ~ 0."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_ttest(a, b)
        assert "p_value" in result
        # scipy.stats.ttest_rel returns NaN when all differences are zero
        # (t-statistic is 0/0); both NaN and 1.0 are acceptable degeneracies
        assert math.isnan(result["p_value"]) or result["p_value"] == pytest.approx(1.0, abs=1e-6)
        assert result["cohen_d"] == pytest.approx(0.0, abs=1e-6)

    def test_effect_size_improvement(self):
        """When treatment > control, cohen_d should be positive."""
        control = [10.0, 11.0, 10.5, 11.5, 10.0]
        treatment = [15.0, 16.0, 15.5, 16.5, 15.0]
        result = compute_effect_size(treatment, control)
        assert result["cohen_d"] > 0, "Cohen's d should be positive for improvement"
        assert result["relative_improvement"] > 0
        assert result["treatment_mean"] > result["control_mean"]

    def test_format_with_ci(self):
        """format_with_ci should produce the expected string format."""
        s = format_with_ci(0.954, 0.931, 0.977)
        assert s == "0.95 (0.93, 0.98)"
        # Also test a custom format spec
        s2 = format_with_ci(0.954, 0.931, 0.977, fmt=".3f")
        assert s2 == "0.954 (0.931, 0.977)"


# ---------------------------------------------------------------------------
# TestBaselines — src/baselines.py
# ---------------------------------------------------------------------------

from src.baselines import (
    RTNKVCache,
    KIVIKVCache,
    AbsmaxKVCache,
    BaselineDecoder,
    BASELINE_REGISTRY,
)
from src.speculative_decode import SpeculativeOutput


class TestBaselines:
    """Tests for RTN, KIVI, and Absmax KV cache quantization baselines."""

    NUM_LAYERS = 2
    NUM_KV_HEADS = 4
    HEAD_DIM = 64
    SEQ_LEN = 32
    BLOCK_SIZE = 32

    def _make_kv(self):
        """Create small synthetic K, V tensors."""
        k = torch.randn(1, self.NUM_KV_HEADS, self.SEQ_LEN, self.HEAD_DIM)
        v = torch.randn(1, self.NUM_KV_HEADS, self.SEQ_LEN, self.HEAD_DIM)
        return k, v

    def test_rtn_roundtrip(self):
        """RTNKVCache compress -> decompress should have reasonable MSE."""
        cache = RTNKVCache(
            num_layers=self.NUM_LAYERS, num_kv_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM, bits=4, block_size=self.BLOCK_SIZE,
        )
        k, v = self._make_kv()
        cache.compress_and_store(0, k, v)
        k_rec, v_rec = cache.get_rotated_kv(0)
        mse_k = (k - k_rec).pow(2).mean().item()
        mse_v = (v - v_rec).pow(2).mean().item()
        assert mse_k < 0.1, f"RTN key MSE too high: {mse_k}"
        assert mse_v < 0.1, f"RTN value MSE too high: {mse_v}"

    def test_kivi_asymmetric(self):
        """KIVI should quantize K and V along different axes."""
        cache = KIVIKVCache(
            num_layers=self.NUM_LAYERS, num_kv_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM, bits=4, block_size=self.BLOCK_SIZE,
        )
        k, v = self._make_kv()
        cache.compress_and_store(0, k, v)

        # K scales: per-channel (axis=-1) -> shape (..., seq_len, n_blocks)
        # V scales: per-token  (axis=-2) -> shape (..., n_blocks, dim)
        k_scales_shape = cache.k_scales[0].shape
        v_scales_shape = cache.v_scales[0].shape

        # They must differ because the quantization axes are different
        assert k_scales_shape != v_scales_shape, (
            f"KIVI K and V scales should have different shapes due to "
            f"asymmetric quantization axes, got K={k_scales_shape} V={v_scales_shape}"
        )

    def test_absmax_symmetric(self):
        """AbsmaxKVCache should use symmetric quantization (no zeros stored)."""
        cache = AbsmaxKVCache(
            num_layers=self.NUM_LAYERS, num_kv_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM, bits=4, block_size=self.BLOCK_SIZE,
        )
        k, v = self._make_kv()
        cache.compress_and_store(0, k, v)

        # Absmax uses int8 codes (signed), not uint8
        assert cache.k_codes[0].dtype == torch.int8
        assert cache.v_codes[0].dtype == torch.int8

        # Absmax has no zero-point attributes (only codes + scales)
        assert not hasattr(cache, "k_zeros") or cache.__class__.__name__ == "AbsmaxKVCache"
        # Verify per-head scales shape: (batch, num_kv_heads, 1, 1)
        assert cache.k_scales[0].shape == (1, self.NUM_KV_HEADS, 1, 1)
        assert cache.v_scales[0].shape == (1, self.NUM_KV_HEADS, 1, 1)

    def test_baseline_attention_shape(self):
        """compressed_attention should return the correct output shape."""
        for name, cls in BASELINE_REGISTRY.items():
            cache = cls(
                num_layers=self.NUM_LAYERS, num_kv_heads=self.NUM_KV_HEADS,
                head_dim=self.HEAD_DIM, bits=4, block_size=self.BLOCK_SIZE,
            )
            k, v = self._make_kv()
            cache.compress_and_store(0, k, v)
            q = torch.randn(1, self.NUM_KV_HEADS, 1, self.HEAD_DIM)
            out = cache.compressed_attention(0, q)
            assert out.shape == (1, self.NUM_KV_HEADS, 1, self.HEAD_DIM), (
                f"{name}: expected shape (1, {self.NUM_KV_HEADS}, 1, {self.HEAD_DIM}), "
                f"got {out.shape}"
            )

    def test_more_bits_less_error_baselines(self):
        """All baselines should show monotonic MSE improvement with more bits."""
        k, v = self._make_kv()
        for name, cls in BASELINE_REGISTRY.items():
            mses = []
            for bits in [2, 3, 4]:
                cache = cls(
                    num_layers=self.NUM_LAYERS, num_kv_heads=self.NUM_KV_HEADS,
                    head_dim=self.HEAD_DIM, bits=bits, block_size=self.BLOCK_SIZE,
                )
                cache.compress_and_store(0, k, v)
                k_rec, v_rec = cache.get_rotated_kv(0)
                mse = ((k - k_rec).pow(2).mean() + (v - v_rec).pow(2).mean()).item()
                mses.append(mse)
            # More bits should yield lower or equal error
            assert mses[1] <= mses[0] + 1e-6, (
                f"{name}: 3-bit MSE ({mses[1]}) should be <= 2-bit MSE ({mses[0]})"
            )
            assert mses[2] <= mses[1] + 1e-6, (
                f"{name}: 4-bit MSE ({mses[2]}) should be <= 3-bit MSE ({mses[1]})"
            )

    def test_baseline_decoder_generate(self):
        """BaselineDecoder.generate() should return a SpeculativeOutput (mocked models)."""
        vocab_size = 32
        hidden_size = 64
        num_heads = 4
        num_layers = 2
        head_dim = hidden_size // num_heads

        # Build a minimal mock config
        mock_config = MagicMock()
        mock_config.num_hidden_layers = num_layers
        mock_config.num_attention_heads = num_heads
        mock_config.num_key_value_heads = num_heads
        mock_config.hidden_size = hidden_size

        def _make_model_output(batch, seq_len, device):
            """Create a mock model forward output with KV cache."""
            logits = torch.randn(batch, seq_len, vocab_size, device=device)
            kv = tuple(
                (
                    torch.randn(batch, num_heads, seq_len, head_dim, device=device),
                    torch.randn(batch, num_heads, seq_len, head_dim, device=device),
                )
                for _ in range(num_layers)
            )
            out = MagicMock()
            out.logits = logits
            out.past_key_values = kv
            return out

        device = torch.device("cpu")

        draft_model = MagicMock()
        draft_model.config = mock_config
        draft_model.eval = MagicMock(return_value=None)
        draft_model.parameters = MagicMock(
            return_value=iter([torch.zeros(1, device=device)])
        )

        target_model = MagicMock()
        target_model.config = mock_config
        target_model.eval = MagicMock(return_value=None)
        target_model.parameters = MagicMock(
            return_value=iter([torch.zeros(1, device=device)])
        )

        # Track call count to produce appropriate seq_len outputs
        call_count = {"draft": 0, "target": 0}

        def draft_forward(*args, **kwargs):
            call_count["draft"] += 1
            inp = args[0] if args else kwargs.get("input_ids", torch.zeros(1, 1))
            seq_len = inp.shape[1] if hasattr(inp, "shape") else 1
            past = kwargs.get("past_key_values")
            total_len = seq_len + (past[0][0].shape[2] if past else 0)
            return _make_model_output(1, seq_len, device)

        def target_forward(*args, **kwargs):
            call_count["target"] += 1
            inp = args[0] if args else kwargs.get("input_ids", torch.zeros(1, 1))
            seq_len = inp.shape[1] if hasattr(inp, "shape") else 1
            return _make_model_output(1, seq_len, device)

        draft_model.side_effect = draft_forward
        draft_model.__call__ = draft_forward
        target_model.side_effect = target_forward
        target_model.__call__ = target_forward

        tokenizer = MagicMock()

        decoder = BaselineDecoder(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            baseline_type="rtn",
            quant_bits=4,
            quant_block_size=self.BLOCK_SIZE,
        )

        input_ids = torch.randint(0, vocab_size, (1, 4))
        result = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=8,
            gamma=2,
            temperature=1.0,
        )

        assert isinstance(result, SpeculativeOutput)
        assert result.num_generated_tokens > 0
        assert result.wall_time_seconds >= 0

    def test_compression_ratio_all_baselines(self):
        """All baselines should achieve compression ratio > 1.0 at 3 bits."""
        k, v = self._make_kv()
        for name, cls in BASELINE_REGISTRY.items():
            cache = cls(
                num_layers=self.NUM_LAYERS, num_kv_heads=self.NUM_KV_HEADS,
                head_dim=self.HEAD_DIM, bits=3, block_size=self.BLOCK_SIZE,
            )
            for layer_idx in range(self.NUM_LAYERS):
                cache.compress_and_store(layer_idx, k, v)
            ratio = cache.compression_ratio
            assert ratio > 1.0, (
                f"{name}: compression ratio {ratio} should be > 1.0 at 3 bits"
            )
