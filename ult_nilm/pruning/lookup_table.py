"""Memory lookup table for hardware-aware subnet pruning.

Implements Eq. (8)-(9) in the paper:

.. math::

    M(\\alpha) = \\sum_{i=1}^{N_\\text{layer}} \\mathcal{T}_M
        (\\alpha_d^i, \\alpha_w^i, \\alpha_k^i)

The table stores per-block memory footprints for every
``(expand_ratio, kernel_size)`` combination, keyed by the position of the
block inside the supernet. Inactive blocks (those beyond the depth chosen
for their stage) do not contribute, which realises the ``alpha_d`` gating.

Memory is approximated as the trainable parameter count (4 bytes per
float) plus a conservative peak-activation estimate for the chosen
configuration. This follows the MCU deployment assumption in Section IV.A:
Flash for parameters, SRAM for the largest intermediate activation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ult_nilm.networks.elastic import NILMSupernet
from ult_nilm.utils.base import BaseNetwork
from ult_nilm.utils.common import make_divisible

BYTES_PER_FLOAT = 4


@dataclass
class BlockMemoryEntry:
    param_bytes: int
    activation_bytes: int

    @property
    def total(self) -> int:
        return self.param_bytes + self.activation_bytes


class MemoryLookupTable:
    """Per-block memory index over the elastic configuration space."""

    def __init__(self, supernet: NILMSupernet, seq_length: int | None = None) -> None:
        self.block_table: dict[tuple[int, int, int], BlockMemoryEntry] = {}
        self.fixed_param_bytes: int = 0
        self.fixed_activation_bytes: int = 0
        self.block_group_info: list[list[int]] = supernet.block_group_info
        self.seq_length = seq_length if seq_length is not None else supernet.seq_length

    # ------------------------------------------------------------ build
    def build_from_supernet(self, supernet: NILMSupernet) -> "MemoryLookupTable":
        self.fixed_param_bytes = self._fixed_param_bytes(supernet)
        self.fixed_activation_bytes = self._fixed_activation_bytes(supernet)

        for block_idx, block in enumerate(supernet.blocks):
            if block_idx == 0:
                # First ``NILMResidualBlock`` is structurally static.
                entry = self._block_entry_static(block, supernet)
                self.block_table[(0, 1, block.conv.kernel_size)] = entry
                continue
            dyn_conv = block.conv
            for expand in dyn_conv.expand_ratio_list:
                for kernel in dyn_conv.kernel_size_list:
                    self.block_table[(block_idx, expand, kernel)] = self._block_entry(
                        dyn_conv.in_channels, dyn_conv.out_channels, expand, kernel,
                        dyn_conv.stride, block.enable_fusion,
                    )
        return self

    # ------------------------------------------------------------ lookup
    def lookup(
        self,
        ks_list: list[int],
        expand_list: list[int],
        depth_list: list[int],
        reduction: str = "sum",
    ) -> int:
        """Return total memory (in bytes) for a given elastic configuration.

        ``ks_list`` and ``expand_list`` are indexed over
        ``supernet.blocks[1:]`` as produced by
        :meth:`NILMSupernet.sample_active_subnet`. ``depth_list`` contains
        the per-stage depth choices that gate which blocks are active.

        ``reduction="sum"`` combines parameter and peak-activation memory;
        ``reduction="param"`` returns Flash usage only; ``reduction="sram"``
        returns the peak SRAM estimate.
        """
        param = self.fixed_param_bytes
        peak_activation = self.fixed_activation_bytes
        first_block = next(iter(k for k in self.block_table if k[0] == 0))
        param += self.block_table[first_block].param_bytes
        peak_activation = max(peak_activation, self.block_table[first_block].activation_bytes)

        for stage_id, depth in enumerate(depth_list):
            active_blocks = self.block_group_info[stage_id][:depth]
            for block_idx in active_blocks:
                choice = (block_idx, expand_list[block_idx - 1], ks_list[block_idx - 1])
                if choice not in self.block_table:
                    raise KeyError(f"missing lookup entry for {choice}")
                entry = self.block_table[choice]
                param += entry.param_bytes
                peak_activation = max(peak_activation, entry.activation_bytes)

        if reduction == "param":
            return param
        if reduction == "sram":
            return peak_activation
        if reduction == "sum":
            return param + peak_activation
        raise ValueError(f"unsupported reduction: {reduction}")

    def lookup_from_sample(self, sample: dict, reduction: str = "sum") -> int:
        """Convenience wrapper accepting a ``{ks, e, d}`` dict."""
        return self.lookup(sample["ks"], sample["e"], sample["d"], reduction=reduction)

    # ------------------------------------------------------------ io
    def save(self, path: str) -> None:
        payload = {
            "fixed_param_bytes": self.fixed_param_bytes,
            "fixed_activation_bytes": self.fixed_activation_bytes,
            "block_group_info": self.block_group_info,
            "seq_length": self.seq_length,
            "table": {
                f"{key[0]},{key[1]},{key[2]}": {
                    "param_bytes": entry.param_bytes,
                    "activation_bytes": entry.activation_bytes,
                }
                for key, entry in self.block_table.items()
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str) -> "MemoryLookupTable":
        with open(path, "r") as f:
            payload = json.load(f)
        self.fixed_param_bytes = payload["fixed_param_bytes"]
        self.fixed_activation_bytes = payload["fixed_activation_bytes"]
        self.block_group_info = payload["block_group_info"]
        self.seq_length = payload["seq_length"]
        self.block_table = {
            tuple(int(k) for k in key.split(",")): BlockMemoryEntry(**entry)
            for key, entry in payload["table"].items()
        }
        return self

    # --------------------------------------------------------- internals
    def _block_entry(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        enable_fusion: bool,
    ) -> BlockMemoryEntry:
        mid = make_divisible(in_channels * expand_ratio, BaseNetwork.CHANNEL_DIVISIBLE)
        expand_params = (in_channels * mid + mid * 2) if mid != in_channels else 0
        dw_params = mid * kernel_size * kernel_size
        pw_params = mid * out_channels
        bn_params = (mid + out_channels) * 2

        transfer_dw = mid * 3 * 3
        transfer_pw = mid * mid
        transfer_bn = mid * 4
        fusion_params = (mid * 2 + 2 * 2) if enable_fusion else 0
        total_params = (
            expand_params + dw_params + pw_params + bn_params
            + transfer_dw + transfer_pw + transfer_bn + fusion_params
        )

        activation_elems = max(mid, out_channels) * self.seq_length
        return BlockMemoryEntry(
            param_bytes=total_params * BYTES_PER_FLOAT,
            activation_bytes=activation_elems * BYTES_PER_FLOAT,
        )

    def _block_entry_static(self, block, supernet: NILMSupernet) -> BlockMemoryEntry:
        conv = block.conv
        mid = conv.in_channels
        out_channels = conv.out_channels
        kernel = conv.kernel_size
        dw_params = mid * kernel * kernel
        pw_params = mid * out_channels
        bn_params = (mid + out_channels) * 2
        total_params = dw_params + pw_params + bn_params
        activation_elems = out_channels * self.seq_length
        return BlockMemoryEntry(
            param_bytes=total_params * BYTES_PER_FLOAT,
            activation_bytes=activation_elems * BYTES_PER_FLOAT,
        )

    def _fixed_param_bytes(self, supernet: NILMSupernet) -> int:
        """Sum of parameter bytes for layers outside the elastic search."""
        total = 0
        for module in (supernet.first_conv, supernet.feature_mix_layer, supernet.classifier):
            total += sum(p.numel() for p in module.parameters() if p.requires_grad) * BYTES_PER_FLOAT
        if hasattr(supernet, "freq_feature_extractor"):
            total += (
                sum(p.numel() for p in supernet.freq_feature_extractor.parameters() if p.requires_grad)
                * BYTES_PER_FLOAT
            )
        if hasattr(supernet, "seq_decoder"):
            total += (
                sum(p.numel() for p in supernet.seq_decoder.parameters() if p.requires_grad)
                * BYTES_PER_FLOAT
            )
        return total

    def _fixed_activation_bytes(self, supernet: NILMSupernet) -> int:
        """Peak-activation estimate for layers outside the elastic search."""
        first_conv_out = supernet.first_conv.out_channels * self.seq_length
        mix_out = supernet.feature_mix_layer.out_channels * self.seq_length
        return max(first_conv_out, mix_out) * BYTES_PER_FLOAT
