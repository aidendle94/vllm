# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TurboQuant KV cache quantization.

Validates the core algorithm from:
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al., 2025)
"""

import math

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant import (
    LLOYD_MAX_BOUNDARIES,
    LLOYD_MAX_CENTROIDS,
    TURBOQUANT_DEFAULT_BITS,
    TURBOQUANT_SUPPORTED_BITS,
    TurboQuantState,
    fast_hadamard_transform,
    pack_1bit,
    pack_2bit,
    pack_3bit,
    pack_4bit,
    packed_head_size,
    parse_turboquant_dtype,
    unpack_1bit,
    unpack_2bit,
    unpack_3bit,
    unpack_4bit,
)


# ---- Hadamard Transform Tests ----


class TestHadamardTransform:
    def test_basic_4d(self):
        """Test Hadamard on a simple 4-element vector."""
        x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        hx = fast_hadamard_transform(x)
        # H_4 * [1,0,0,0]^T = [1,1,1,1]^T
        torch.testing.assert_close(hx, torch.tensor([1.0, 1.0, 1.0, 1.0]))

    def test_norm_scaling(self):
        """||H·x|| = sqrt(d) · ||x|| for Hadamard."""
        d = 128
        x = torch.randn(d)
        hx = fast_hadamard_transform(x)
        expected_norm = math.sqrt(d) * x.norm()
        assert abs(hx.norm().item() - expected_norm.item()) < 1e-3

    def test_involution(self):
        """H · H · x = d · x (Hadamard is an involution up to scaling)."""
        d = 64
        x = torch.randn(d)
        hhx = fast_hadamard_transform(fast_hadamard_transform(x))
        torch.testing.assert_close(hhx, d * x, atol=1e-4, rtol=1e-4)

    def test_batched(self):
        """Test batched Hadamard transform."""
        batch = 8
        d = 32
        x = torch.randn(batch, d)
        hx = fast_hadamard_transform(x)
        for i in range(batch):
            expected = fast_hadamard_transform(x[i])
            torch.testing.assert_close(hx[i], expected)

    def test_3d_batched(self):
        """Test 3D batched Hadamard (num_tokens, num_heads, head_dim)."""
        x = torch.randn(4, 8, 128)
        hx = fast_hadamard_transform(x)
        assert hx.shape == (4, 8, 128)
        expected = fast_hadamard_transform(x[2, 3])
        torch.testing.assert_close(hx[2, 3], expected)


# ---- Packing Tests ----


class TestPacking:
    def test_pack_unpack_1bit_roundtrip(self):
        head_dim = 128
        indices = torch.randint(0, 2, (16, 8, head_dim), dtype=torch.uint8)
        packed = pack_1bit(indices)
        assert packed.shape == (16, 8, head_dim // 8)
        unpacked = unpack_1bit(packed, head_dim)
        torch.testing.assert_close(unpacked, indices)

    def test_pack_unpack_2bit_roundtrip(self):
        head_dim = 128
        indices = torch.randint(0, 4, (16, 8, head_dim), dtype=torch.uint8)
        packed = pack_2bit(indices)
        assert packed.shape == (16, 8, head_dim // 4)
        unpacked = unpack_2bit(packed, head_dim)
        torch.testing.assert_close(unpacked, indices)

    def test_pack_unpack_3bit_roundtrip(self):
        head_dim = 128
        indices = torch.randint(0, 8, (16, 8, head_dim), dtype=torch.uint8)
        packed = pack_3bit(indices)
        # 128 values, 8 per 3 bytes → 128/8*3 = 48 bytes
        assert packed.shape == (16, 8, 48)
        unpacked = unpack_3bit(packed, head_dim)
        torch.testing.assert_close(unpacked, indices)

    def test_pack_unpack_4bit_roundtrip(self):
        head_dim = 128
        indices = torch.randint(0, 16, (16, 8, head_dim), dtype=torch.uint8)
        packed = pack_4bit(indices)
        assert packed.shape == (16, 8, head_dim // 2)
        unpacked = unpack_4bit(packed, head_dim)
        torch.testing.assert_close(unpacked, indices)

    def test_packed_head_size(self):
        assert packed_head_size(128, 1) == 18   # 16 + 2
        assert packed_head_size(128, 2) == 34   # 32 + 2
        assert packed_head_size(128, 3) == 50   # 48 + 2
        assert packed_head_size(128, 4) == 66   # 64 + 2
        assert packed_head_size(64, 1) == 10    # 8 + 2
        assert packed_head_size(64, 2) == 18    # 16 + 2
        assert packed_head_size(64, 3) == 26    # 24 + 2
        assert packed_head_size(64, 4) == 34    # 32 + 2


# ---- parse_turboquant_dtype Tests ----


class TestParseDtype:
    def test_default(self):
        assert parse_turboquant_dtype("turboquant") == TURBOQUANT_DEFAULT_BITS
        assert parse_turboquant_dtype("turboquant") == 4

    def test_explicit_variants(self):
        assert parse_turboquant_dtype("turboquant_1bit") == 1
        assert parse_turboquant_dtype("turboquant_2bit") == 2
        assert parse_turboquant_dtype("turboquant_3bit") == 3
        assert parse_turboquant_dtype("turboquant_4bit") == 4

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_turboquant_dtype("turboquant_5bit")
        with pytest.raises(ValueError):
            parse_turboquant_dtype("fp8")


# ---- TurboQuantState Tests ----


class TestTurboQuantState:
    @pytest.fixture(params=[1, 2, 3, 4])
    def tq_state(self, request):
        return TurboQuantState(
            head_dim=128, bit_width=request.param, seed=42,
            device="cpu", dtype=torch.float32,
        )

    @pytest.fixture
    def tq_state_2bit(self):
        return TurboQuantState(
            head_dim=128, bit_width=2, seed=42, device="cpu", dtype=torch.float32
        )

    @pytest.fixture
    def tq_state_4bit(self):
        return TurboQuantState(
            head_dim=128, bit_width=4, seed=42, device="cpu", dtype=torch.float32
        )

    def test_rotation_inverse(self, tq_state):
        """rotate then inverse_rotate should recover the original vector."""
        x = torch.randn(4, 8, 128)
        z = tq_state.rotate(x)
        x_recovered = tq_state.inverse_rotate(z)
        torch.testing.assert_close(x_recovered, x, atol=1e-4, rtol=1e-4)

    def test_rotation_preserves_inner_product(self, tq_state_2bit):
        """Inner product is preserved under rotation (orthogonal transform)."""
        tq = tq_state_2bit
        a = torch.randn(128)
        b = torch.randn(128)
        ip_original = (a * b).sum()
        a_rot = tq.rotate(a)
        b_rot = tq.rotate(b)
        ip_rotated = (a_rot * b_rot).sum()
        # <H·D·a, H·D·b> = d · <a, b>
        torch.testing.assert_close(
            ip_rotated / 128.0, ip_original, atol=1e-4, rtol=1e-4
        )

    def test_quantize_dequant_shape(self, tq_state):
        """Quantize → dequant should produce correct shapes for all bit widths."""
        x = torch.randn(16, 8, 128)
        packed = tq_state.quantize(x)
        expected_size = packed_head_size(128, tq_state.bit_width)
        assert packed.shape == (16, 8, expected_size)
        assert packed.dtype == torch.uint8

        deq = tq_state.dequantize(packed, output_dtype=torch.float32)
        assert deq.shape == (16, 8, 128)

    def test_quantize_dequant_preserves_norm(self, tq_state):
        """Norm should be approximately preserved for all bit widths."""
        x = torch.randn(100, 128) * 3.0
        packed = tq_state.quantize(x)
        deq = tq_state.dequantize(packed, output_dtype=torch.float32)

        original_norms = x.norm(dim=-1)
        deq_norms = deq.norm(dim=-1)
        ratio = deq_norms / (original_norms + 1e-10)
        assert ratio.mean().item() > 0.3, (
            f"Mean norm ratio too low: {ratio.mean()} for {tq_state.bit_width}-bit"
        )

    def test_mse_decreases_with_bits(self):
        """Higher bit widths should yield lower MSE."""
        d = 128
        n = 500
        torch.manual_seed(123)
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)

        mse_values = {}
        for b in TURBOQUANT_SUPPORTED_BITS:
            tq = TurboQuantState(
                head_dim=d, bit_width=b, seed=42,
                device="cpu", dtype=torch.float32,
            )
            packed = tq.quantize(x)
            deq_rot = tq.dequantize(packed, output_dtype=torch.float32)
            deq = tq.inverse_rotate(deq_rot)
            mse_values[b] = ((x - deq) ** 2).sum(dim=-1).mean().item()

        # MSE should strictly decrease with more bits
        for i in range(len(TURBOQUANT_SUPPORTED_BITS) - 1):
            b_low = TURBOQUANT_SUPPORTED_BITS[i]
            b_high = TURBOQUANT_SUPPORTED_BITS[i + 1]
            assert mse_values[b_low] > mse_values[b_high], (
                f"MSE did not decrease: {b_low}-bit={mse_values[b_low]:.4f} "
                f"vs {b_high}-bit={mse_values[b_high]:.4f}"
            )

    def test_mse_distortion_2bit(self, tq_state_2bit):
        """MSE distortion for 2-bit. Paper says D_mse ≈ 0.117."""
        tq = tq_state_2bit
        d = 128
        n = 1000
        torch.manual_seed(123)
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)
        packed = tq.quantize(x)
        deq_rot = tq.dequantize(packed, output_dtype=torch.float32)
        deq = tq.inverse_rotate(deq_rot)
        mse = ((x - deq) ** 2).sum(dim=-1).mean().item()
        assert mse < 0.5, f"MSE {mse} is too high for 2-bit quantization"

    def test_mse_distortion_4bit(self, tq_state_4bit):
        """4-bit should have much lower MSE. Paper says D_mse ≈ 0.009."""
        tq = tq_state_4bit
        d = 128
        n = 1000
        torch.manual_seed(123)
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)
        packed = tq.quantize(x)
        deq_rot = tq.dequantize(packed, output_dtype=torch.float32)
        deq = tq.inverse_rotate(deq_rot)
        mse = ((x - deq) ** 2).sum(dim=-1).mean().item()
        assert mse < 0.1, f"MSE {mse} is too high for 4-bit quantization"

    def test_inner_product_preservation(self, tq_state_2bit):
        """Inner products should be approximately preserved."""
        tq = tq_state_2bit
        d = 128
        torch.manual_seed(456)
        q = torch.randn(50, d)
        k = torch.randn(200, d)

        true_ip = q @ k.T
        packed_k = tq.quantize(k)
        k_rot = tq.dequantize(packed_k, output_dtype=torch.float32)
        q_rot = tq.rotate(q)

        approx_ip = q_rot @ k_rot.T

        true_flat = (d * true_ip).flatten()
        approx_flat = approx_ip.flatten()
        correlation = torch.corrcoef(torch.stack([true_flat, approx_flat]))[0, 1]
        assert correlation > 0.9, (
            f"Inner product correlation {correlation} too low"
        )

    def test_cache_quantize_scatter(self, tq_state):
        """Test quantize_and_scatter for all bit widths."""
        tq = tq_state
        num_blocks, block_size, num_kv_heads, head_dim = 4, 16, 8, 128

        ps = tq.packed_head_size
        kv_cache = torch.zeros(
            num_blocks, 2, block_size, num_kv_heads, ps, dtype=torch.uint8
        )

        num_tokens = 5
        key = torch.randn(num_tokens, num_kv_heads, head_dim)
        value = torch.randn(num_tokens, num_kv_heads, head_dim)
        slot_mapping = torch.tensor([0, 1, 16, 17, 32], dtype=torch.int64)

        tq.quantize_and_scatter(key, value, kv_cache, slot_mapping)

        assert kv_cache[0, 0, 0].any(), "Key slot should have data"
        assert kv_cache[0, 1, 0].any(), "Value slot should have data"
        assert not kv_cache[3, 0, 0].any(), "Unused slot should be zero"

    def test_cache_dequantize_roundtrip(self, tq_state):
        """Test full cache quantize → dequantize roundtrip for all bit widths."""
        tq = tq_state
        num_blocks, block_size, num_kv_heads, head_dim = 2, 16, 4, 128

        ps = tq.packed_head_size
        kv_cache = torch.zeros(
            num_blocks, 2, block_size, num_kv_heads, ps, dtype=torch.uint8
        )

        num_tokens = 32
        key = torch.randn(num_tokens, num_kv_heads, head_dim)
        value = torch.randn(num_tokens, num_kv_heads, head_dim)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64)

        tq.quantize_and_scatter(key, value, kv_cache, slot_mapping)

        key_rot, value_rot = tq.dequantize_cache(
            kv_cache, output_dtype=torch.float32
        )
        assert key_rot.shape == (num_blocks, block_size, num_kv_heads, head_dim)
        assert value_rot.shape == (num_blocks, block_size, num_kv_heads, head_dim)


# ---- Codebook Tests ----


class TestCodebook:
    def test_centroids_are_symmetric(self):
        for b, centroids in LLOYD_MAX_CENTROIDS.items():
            n = len(centroids)
            for i in range(n // 2):
                assert abs(centroids[i] + centroids[n - 1 - i]) < 1e-3, (
                    f"Centroid {i} and {n-1-i} not symmetric for b={b}"
                )

    def test_centroids_are_sorted(self):
        for b, centroids in LLOYD_MAX_CENTROIDS.items():
            for i in range(len(centroids) - 1):
                assert centroids[i] < centroids[i + 1]

    def test_centroids_count(self):
        for b, centroids in LLOYD_MAX_CENTROIDS.items():
            assert len(centroids) == 2**b

    def test_boundaries_count(self):
        for b in LLOYD_MAX_CENTROIDS:
            assert len(LLOYD_MAX_BOUNDARIES[b]) == len(LLOYD_MAX_CENTROIDS[b]) - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
