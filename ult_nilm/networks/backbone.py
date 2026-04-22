"""Static container used as the NILM supernet's structural base.

``NILMBackbone`` mirrors the public API of the elastic network so a subnet
exported via ``get_active_subnet`` can be rehydrated with the same
configuration-based loader.
"""

from __future__ import annotations

import torch.nn as nn

from ult_nilm.modules.layers import (
    IdentityLayer,
    MBConvLayer,
    ResidualBlock,
    set_layer_from_config,
)
from ult_nilm.utils.base import BaseNetwork


class _GlobalAvgPool2d(nn.Module):
    def __init__(self, keep_dim: bool = False):
        super().__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)


class NILMBackbone(BaseNetwork):
    """Static block-structured network used as the NILM supernet scaffold."""

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super().__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pool = _GlobalAvgPool2d(keep_dim=False)
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self) -> str:
        parts = [self.first_conv.module_str]
        parts.extend(block.module_str for block in self.blocks)
        if self.feature_mix_layer is not None:
            parts.append(self.feature_mix_layer.module_str)
        parts.append(repr(self.global_avg_pool))
        parts.append(self.classifier.module_str)
        return "\n".join(parts)

    @property
    def config(self) -> dict:
        return {
            "name": NILMBackbone.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "feature_mix_layer": None
            if self.feature_mix_layer is None
            else self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config: dict) -> "NILMBackbone":
        first_conv = set_layer_from_config(config["first_conv"])
        feature_mix_layer = set_layer_from_config(config["feature_mix_layer"])
        classifier = set_layer_from_config(config["classifier"])
        blocks = [ResidualBlock.build_from_config(block_config) for block_config in config["blocks"]]
        net = NILMBackbone(first_conv, blocks, feature_mix_layer, classifier)
        if "bn" in config and config["bn"] is not None:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)
        return net

    def zero_last_gamma(self) -> None:
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                if isinstance(m.conv, MBConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.conv.point_linear.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list: list[list[int]] = []
        block_index_list: list[int] = []
        for i, block in enumerate(self.blocks[1:], 1):
            if block.shortcut is None and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def load_state_dict(self, state_dict, **kwargs):
        current_state_dict = self.state_dict()
        for key in state_dict:
            if key not in current_state_dict:
                assert ".mobile_inverted_conv." in key
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            current_state_dict[new_key] = state_dict[key]
        super().load_state_dict(current_state_dict)
