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
