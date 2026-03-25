# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant: Online Vector Quantization for KV Cache Compression.

Implements the TurboQuant algorithm from:
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al., 2025, arXiv:2504.19874)

TurboQuant compresses KV cache vectors by:
1. Applying a random rotation (fast Walsh-Hadamard transform with random
   sign flips) to normalize the coordinate distribution.
2. Quantizing each coordinate independently using precomputed optimal
   Lloyd-Max scalar quantizers for the resulting near-Gaussian distribution.
3. Packing quantized indices + per-vector norms into compact uint8 storage.

For attention, the computation is done in "rotated space" to avoid
inverse-rotating the entire cache on every read:
- Q is rotated before attention.
- K/V are dequantized directly in rotated space (just codebook lookup + norm).
- The attention output is inverse-rotated back to original space.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Lloyd-Max optimal centroids for N(0, 1)
#
# These are precomputed centroids that minimize E[(X - Q(X))^2] where
# X ~ N(0,1) and Q is a b-bit scalar quantizer. After random rotation,
# each coordinate of a unit-norm vector follows approx N(0, 1/d), so we
# scale these centroids by 1/sqrt(d) at runtime.
#
# Sources: Max (1960), Lloyd (1982), standard quantization tables.
# ---------------------------------------------------------------------------
LLOYD_MAX_CENTROIDS: dict[int, list[float]] = {
    1: [
        -0.7978845608028654,
        0.7978845608028654,
    ],
    2: [
        -1.510417608880236,
        -0.4527800398860679,
        0.4527800398860679,
        1.510417608880236,
    ],
    3: [
        -2.1519775877,
        -1.3439653776,
        -0.7560054023,
        -0.2451209526,
        0.2451209526,
        0.7560054023,
        1.3439653776,
        2.1519775877,
    ],
    4: [
        -2.7326,
        -2.0690,
        -1.6180,
        -1.2562,
        -0.9424,
        -0.6568,
        -0.3882,
        -0.1284,
        0.1284,
        0.3882,
        0.6568,
        0.9424,
        1.2562,
        1.6180,
        2.0690,
        2.7326,
    ],
}

# Decision boundaries: midpoints between consecutive centroids.
# Used for fast quantization via bucketize.
LLOYD_MAX_BOUNDARIES: dict[int, list[float]] = {}
for _b, _centroids in LLOYD_MAX_CENTROIDS.items():
    _bounds = []
    for i in range(len(_centroids) - 1):
        _bounds.append((_centroids[i] + _centroids[i + 1]) / 2.0)
    LLOYD_MAX_BOUNDARIES[_b] = _bounds

TURBOQUANT_SUPPORTED_BITS = (1, 2, 3, 4)
TURBOQUANT_DEFAULT_BITS = 4


def parse_turboquant_dtype(dtype_str: str) -> int:
    """Parse a TurboQuant cache dtype string and return the bit width.

    Accepted formats:
        "turboquant"       → default (4-bit)
        "turboquant_1bit"  → 1-bit
        "turboquant_2bit"  → 2-bit
        "turboquant_3bit"  → 3-bit
        "turboquant_4bit"  → 4-bit
    """
    if dtype_str == "turboquant":
        return TURBOQUANT_DEFAULT_BITS
    for b in TURBOQUANT_SUPPORTED_BITS:
        if dtype_str == f"turboquant_{b}bit":
            return b
    raise ValueError(
        f"Unknown TurboQuant dtype '{dtype_str}'. "
        f"Supported: 'turboquant' (default {TURBOQUANT_DEFAULT_BITS}-bit), "
        + ", ".join(f"'turboquant_{b}bit'" for b in TURBOQUANT_SUPPORTED_BITS)
    )


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform
# ---------------------------------------------------------------------------
def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply the (unnormalized) Walsh-Hadamard transform along the last dim.

    For input x of shape [..., d] where d is a power of 2, computes H·x
    where H is the d×d Hadamard matrix satisfying H·H^T = d·I.

    After transform, ||H·x|| = sqrt(d) · ||x||.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, (
        f"Last dimension must be a power of 2, got {d}"
    )
    h = 1
    while h < d:
        # Reshape into blocks of size 2h, split into halves, butterfly
        x = x.unflatten(-1, (-1, 2 * h))  # [..., d/(2h), 2h]
        a = x[..., :h]  # first half
        b = x[..., h:]  # second half
        x = torch.cat([a + b, a - b], dim=-1)  # butterfly
        x = x.flatten(-2, -1)  # [..., d]
        h *= 2
    return x


# ---------------------------------------------------------------------------
# Bit-packing utilities
# ---------------------------------------------------------------------------
def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack tensor of 2-bit values (0-3) into uint8, 4 values per byte."""
    d = indices.shape[-1]
    assert d % 4 == 0, f"Last dim must be divisible by 4 for 2-bit packing, got {d}"
    idx = indices.to(torch.uint8).unflatten(-1, (d // 4, 4))
    packed = idx[..., 0] | (idx[..., 1] << 2) | (idx[..., 2] << 4) | (idx[..., 3] << 6)
    return packed


def unpack_2bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Unpack uint8 to 2-bit indices, 4 values per byte."""
    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03
    return torch.stack([b0, b1, b2, b3], dim=-1).flatten(-2, -1)[..., :head_dim]


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack tensor of 4-bit values (0-15) into uint8, 2 values per byte."""
    d = indices.shape[-1]
    assert d % 2 == 0, f"Last dim must be even for 4-bit packing, got {d}"
    idx = indices.to(torch.uint8).unflatten(-1, (d // 2, 2))
    packed = idx[..., 0] | (idx[..., 1] << 4)
    return packed


def unpack_4bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Unpack uint8 to 4-bit indices, 2 values per byte."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).flatten(-2, -1)[..., :head_dim]


def pack_1bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack tensor of 1-bit values (0-1) into uint8, 8 values per byte."""
    d = indices.shape[-1]
    assert d % 8 == 0, f"Last dim must be divisible by 8 for 1-bit packing, got {d}"
    idx = indices.to(torch.uint8).unflatten(-1, (d // 8, 8))
    packed = torch.zeros_like(idx[..., 0])
    for i in range(8):
        packed = packed | (idx[..., i] << i)
    return packed


def unpack_1bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Unpack uint8 to 1-bit indices, 8 values per byte."""
    bits = [(packed >> i) & 0x01 for i in range(8)]
    return torch.stack(bits, dim=-1).flatten(-2, -1)[..., :head_dim]


def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack tensor of 3-bit values (0-7) into uint8.

    Uses a 3-bytes-per-8-values scheme: 8 values × 3 bits = 24 bits = 3 bytes.
    """
    d = indices.shape[-1]
    assert d % 8 == 0, f"Last dim must be divisible by 8 for 3-bit packing, got {d}"
    idx = indices.to(torch.uint16).unflatten(-1, (d // 8, 8))
    # Pack 8 × 3-bit values into 3 bytes (24 bits)
    # Concatenate all 24 bits, then split into 3 bytes
    packed_24 = torch.zeros(
        *idx.shape[:-1], dtype=torch.int32, device=idx.device
    )
    for i in range(8):
        packed_24 = packed_24 | (idx[..., i].to(torch.int32) << (i * 3))
    b0 = (packed_24 & 0xFF).to(torch.uint8)
    b1 = ((packed_24 >> 8) & 0xFF).to(torch.uint8)
    b2 = ((packed_24 >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).flatten(-2, -1)


def unpack_3bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Unpack uint8 to 3-bit indices, 8 values per 3 bytes."""
    d_packed = packed.shape[-1]
    packed = packed.unflatten(-1, (d_packed // 3, 3))
    b0 = packed[..., 0].to(torch.int32)
    b1 = packed[..., 1].to(torch.int32)
    b2 = packed[..., 2].to(torch.int32)
    packed_24 = b0 | (b1 << 8) | (b2 << 16)
    vals = [(packed_24 >> (i * 3)) & 0x07 for i in range(8)]
    result = torch.stack(vals, dim=-1).flatten(-2, -1)[..., :head_dim]
    return result.to(torch.uint8)


def pack_indices(indices: torch.Tensor, bit_width: int) -> torch.Tensor:
    if bit_width == 1:
        return pack_1bit(indices)
    elif bit_width == 2:
        return pack_2bit(indices)
    elif bit_width == 3:
        return pack_3bit(indices)
    elif bit_width == 4:
        return pack_4bit(indices)
    else:
        raise ValueError(
            f"Unsupported bit_width={bit_width}, must be in {TURBOQUANT_SUPPORTED_BITS}"
        )


def unpack_indices(packed: torch.Tensor, bit_width: int, head_dim: int) -> torch.Tensor:
    if bit_width == 1:
        return unpack_1bit(packed, head_dim)
    elif bit_width == 2:
        return unpack_2bit(packed, head_dim)
    elif bit_width == 3:
        return unpack_3bit(packed, head_dim)
    elif bit_width == 4:
        return unpack_4bit(packed, head_dim)
    else:
        raise ValueError(
            f"Unsupported bit_width={bit_width}, must be in {TURBOQUANT_SUPPORTED_BITS}"
        )


def packed_head_size(head_dim: int, bit_width: int) -> int:
    """Compute the number of uint8 bytes needed per head for packed storage.

    Layout: [packed_indices] + [norm (2 bytes as fp16)]

    For bit_width 1, 2, 4: head_dim * bit_width / 8 bytes for indices.
    For bit_width 3: 3 bytes per 8 values → head_dim * 3 / 8 bytes.
    """
    if bit_width == 3:
        assert head_dim % 8 == 0
        idx_bytes = (head_dim // 8) * 3
    else:
        idx_bytes = head_dim * bit_width // 8
    return idx_bytes + 2


# ---------------------------------------------------------------------------
# TurboQuant State: holds rotation parameters and codebook
# ---------------------------------------------------------------------------
class TurboQuantState:
    """Holds the immutable state needed for TurboQuant quantize/dequantize.

    This is created once per model and shared across all layers.

    Attributes:
        head_dim: The attention head dimension (must be power of 2).
        bit_width: Bits per coordinate (2 or 4).
        signs: Random ±1 sign-flip vector of shape [head_dim].
        codebook: Centroid values scaled for this head_dim, shape [2^bit_width].
        boundaries: Decision boundaries for quantization, shape [2^bit_width - 1].
    """

    def __init__(
        self,
        head_dim: int,
        bit_width: int = 2,
        seed: int = 42,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        assert bit_width in TURBOQUANT_SUPPORTED_BITS, (
            f"bit_width must be in {TURBOQUANT_SUPPORTED_BITS}, got {bit_width}"
        )
        assert head_dim > 0 and (head_dim & (head_dim - 1)) == 0, (
            f"head_dim must be a power of 2, got {head_dim}"
        )

        self.head_dim = head_dim
        self.bit_width = bit_width
        self._packed_head_size = packed_head_size(head_dim, bit_width)

        # Generate deterministic random signs for the rotation
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.signs = (
            torch.randint(0, 2, (head_dim,), generator=gen, device="cpu").float() * 2
            - 1
        ).to(device=device, dtype=dtype)

        # Scale centroids: after rotation, coordinates ~ N(0, 1/d),
        # so we multiply N(0,1) centroids by 1/sqrt(d).
        # But we apply the Hadamard transform WITHOUT the 1/sqrt(d) normalization
        # (i.e., unnormalized H), so the rotated coordinates ~ N(0, 1)
        # and we use the raw N(0,1) centroids directly.
        raw_centroids = LLOYD_MAX_CENTROIDS[bit_width]
        self.codebook = torch.tensor(
            raw_centroids, device=device, dtype=dtype
        )

        raw_boundaries = LLOYD_MAX_BOUNDARIES[bit_width]
        self.boundaries = torch.tensor(
            raw_boundaries, device=device, dtype=dtype
        )

    @property
    def packed_head_size(self) -> int:
        return self._packed_head_size

    # ---- Rotation primitives ------------------------------------------------

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation: H · (D · x).

        x: [..., head_dim]
        Returns: [..., head_dim] in rotated space.
        """
        return fast_hadamard_transform(x * self.signs)

    def inverse_rotate(self, z: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: D · (H · z) / d.

        z: [..., head_dim] in rotated space.
        Returns: [..., head_dim] in original space.
        """
        return fast_hadamard_transform(z) * self.signs / self.head_dim

    # ---- Quantize / Dequantize ----------------------------------------------

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors and pack into compact uint8 representation.

        Args:
            x: [..., head_dim] input vectors (keys or values).

        Returns:
            [..., packed_head_size] uint8 tensor containing packed
            quantization indices and fp16 norm.

        Algorithm (per vector):
            1. norm = ||x||₂
            2. x_hat = x / norm  (unit normalize)
            3. z = H · (D · x_hat)  (random rotation, coords ~ N(0,1))
            4. idx = nearest_centroid(z)  (scalar quantization per coord)
            5. pack(idx, norm) → uint8 bytes
        """
        original_shape = x.shape
        head_dim = original_shape[-1]
        assert head_dim == self.head_dim

        # Compute norm and normalize
        norm = x.norm(dim=-1, keepdim=True)  # [..., 1]
        x_hat = x / (norm + 1e-10)

        # Apply rotation
        z = self.rotate(x_hat)  # [..., head_dim], coords ~ N(0, 1)

        # Scalar quantization per coordinate: find nearest centroid
        # Using bucketize for efficiency (boundaries are sorted)
        z_flat = z.reshape(-1, head_dim).to(torch.float32)
        indices = torch.bucketize(z_flat, self.boundaries.to(torch.float32))
        indices = indices.reshape(*original_shape[:-1], head_dim)

        # Pack indices into bytes
        packed_idx = pack_indices(indices, self.bit_width)
        # packed_idx: [..., head_dim * bit_width // 8] uint8

        # Convert norm to fp16 raw bytes
        norm_fp16 = norm.to(torch.float16).contiguous()
        norm_bytes = norm_fp16.view(torch.uint8)  # [..., 2]

        # Concatenate: [packed_indices | norm_bytes]
        result = torch.cat([packed_idx, norm_bytes], dim=-1)
        return result  # [..., packed_head_size] uint8

    def dequantize(
        self, packed: torch.Tensor, output_dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """Dequantize packed representation back to vectors IN ROTATED SPACE.

        The returned vectors are in the rotated coordinate system. To get
        original-space vectors, apply inverse_rotate(). However, for
        attention we operate entirely in rotated space (rotating Q and
        inverse-rotating the output instead).

        Args:
            packed: [..., packed_head_size] uint8 packed data.
            output_dtype: Desired output dtype.

        Returns:
            [..., head_dim] dequantized vectors in rotated space.
        """
        idx_size = self._packed_head_size - 2  # total minus 2 norm bytes

        # Split packed indices and norm
        packed_idx = packed[..., :idx_size]
        norm_bytes = packed[..., idx_size : idx_size + 2].contiguous()
        norms = norm_bytes.view(torch.float16)  # [..., 1]

        # Unpack indices
        indices = unpack_indices(packed_idx, self.bit_width, self.head_dim)

        # Codebook lookup
        centroids = self.codebook[indices.long()]  # [..., head_dim]

        # Scale by norm to get dequantized vector in rotated space
        result = centroids * norms.to(centroids.dtype)

        return result.to(output_dtype)

    def dequantize_cache(
        self,
        compressed_kv: torch.Tensor,
        output_dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize an entire compressed KV cache to fp16 in rotated space.

        Args:
            compressed_kv: [num_blocks, 2, block_size, num_kv_heads,
                            packed_head_size] uint8
            output_dtype: Desired output dtype.

        Returns:
            key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            Both in rotated space.
        """
        key_compressed = compressed_kv[:, 0]   # [num_blocks, block_size, num_kv_heads, packed]
        value_compressed = compressed_kv[:, 1]

        key_dequant = self.dequantize(key_compressed, output_dtype)
        value_dequant = self.dequantize(value_compressed, output_dtype)

        return key_dequant, value_dequant

    def quantize_and_scatter(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Quantize K/V and write to specific slots in the paged cache.

        Args:
            key: [num_tokens, num_kv_heads, head_dim]
            value: [num_tokens, num_kv_heads, head_dim]
            kv_cache: [num_blocks, 2, block_size, num_kv_heads, packed_head_size] uint8
            slot_mapping: [num_tokens] int64, maps each token to a flat cache slot
        """
        if kv_cache.numel() == 0:
            return

        # Quantize
        packed_key = self.quantize(key)    # [num_tokens, num_kv_heads, packed_head_size]
        packed_value = self.quantize(value)

        # Compute block index and offset from slot_mapping
        block_size = kv_cache.shape[2]
        block_idx = slot_mapping // block_size
        block_offset = slot_mapping % block_size

        # Scatter write to cache
        # kv_cache shape: [num_blocks, 2, block_size, num_kv_heads, packed_head_size]
        kv_cache[block_idx, 0, block_offset] = packed_key
        kv_cache[block_idx, 1, block_offset] = packed_value


# ---------------------------------------------------------------------------
# vLLM Quantization Config Integration
# ---------------------------------------------------------------------------
class TurboQuantConfig(QuantizationConfig):
    """Quantization config for TurboQuant KV cache compression."""

    def __init__(self, bit_width: int = TURBOQUANT_DEFAULT_BITS):
        self.bit_width = bit_width

    def __repr__(self) -> str:
        return f"TurboQuantConfig(bit_width={self.bit_width})"

    @classmethod
    def get_name(cls) -> str:
        return "turboquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Works on any GPU with basic tensor ops

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TurboQuantConfig":
        bit_width = cls.get_from_keys(config, ["bit_width"])
        if bit_width is None:
            bit_width = 2
        return cls(bit_width=int(bit_width))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.model_executor.layers.attention.attention import Attention

        if isinstance(layer, Attention):
            return TurboQuantKVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


class TurboQuantKVCacheMethod(QuantizeMethodBase):
    """KV cache quantization method using TurboQuant.

    This method does not add learnable scale parameters (unlike FP8).
    The quantization is data-oblivious and uses a fixed codebook.
    """

    def __init__(self, quant_config: TurboQuantConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        # TurboQuant doesn't use learned scales like FP8.
        # We still register dummy scales for compatibility with the
        # BaseKVCacheMethod interface expectations.
        pass

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(
            f"{self.__class__.__name__}.apply should not be called."
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass
