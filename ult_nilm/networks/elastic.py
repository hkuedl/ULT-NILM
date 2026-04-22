"""NILM elastic supernet.

Implements the depth/width/kernel-elastic network described in Section III
of the paper. The supernet stacks ``DynamicAdaptiveTransferBlock`` units on
top of a ``NILMResidualBlock`` stem; enabling ``use_frequency_features``
activates the parallel frequency branch (Eq. 15-17). The whole network
integrates the domain-adaptation losses so a training script can compute
``forward_domain_adaptation`` end-to-end.
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn as nn

from ult_nilm.losses import CORALLoss, MMDLoss, SinkhornLoss
from ult_nilm.modules.frequency import FrequencyFeatureExtractor
from ult_nilm.modules.layers import (
    ConvLayer,
    LinearLayer,
    MBConvLayer,
    NILMResidualBlock,
    ResidualBlock,
)
from ult_nilm.networks.backbone import NILMBackbone
from ult_nilm.networks.dynamic_layers import (
    DynamicAdaptiveTransferBlock,
    DynamicMBConvLayer,
)
from ult_nilm.utils.base import BaseNetwork
from ult_nilm.utils.common import build_activation, make_divisible, val2list


class NILMSupernet(NILMBackbone):
    """Elastic supernet with depth/width/kernel elasticity.

    Parameters follow Table II of the paper. With the default search space
    (``ks_list=[1,3,5]``, ``expand_ratio_list=[1,3,5]``, ``depth_list=[1,2,3]``)
    the configuration space contains ~5.5e8 unique subnets sharing one set of
    weights.
    """

    def __init__(
        self,
        n_classes: int = 1000,
        bn_param: tuple[float, float] = (0.1, 1e-3),
        dropout_rate: float = 0.1,
        width_mult: float = 1.0,
        ks_list=3,
        expand_ratio_list=6,
        depth_list=4,
        data_channels: int = 3,
        first_stage_kernel_sizes: list[int] = (3, 3),
        first_stage_width: list[int] = (32, 16),
        first_stage_strides: list[int] = (2, 1),
        base_stage_width: list[int] = (24, 40, 80, 96, 192, 320),
        base_stage_strides: list[int] = (2, 2, 2, 1, 2, 1),
        last_stage_width: int = 1280,
        act_func: str = "relu6",
        use_frequency_features: bool = False,
        domain_adaptation_method: str = "sinkhorn_coral",
        seq2seq: bool = False,
        seq_length: int = 1024,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 100,
        sinkhorn_coral_weight: float = 0.6,
        domain_loss_weight: float = 0.3,
    ):
        self.width_mult = width_mult
        self.ks_list = sorted(val2list(ks_list, 1))
        self.expand_ratio_list = sorted(val2list(expand_ratio_list, 1))
        self.depth_list = sorted(val2list(depth_list, 1))

        self.use_frequency_features = use_frequency_features
        self.domain_adaptation_method = domain_adaptation_method
        self.seq2seq = seq2seq
        self.seq_length = seq_length

        input_channel = make_divisible(first_stage_width[0] * width_mult, BaseNetwork.CHANNEL_DIVISIBLE)
        first_block_width = make_divisible(first_stage_width[1] * width_mult, BaseNetwork.CHANNEL_DIVISIBLE)
        last_channel = make_divisible(last_stage_width * width_mult, BaseNetwork.CHANNEL_DIVISIBLE)

        first_conv = ConvLayer(
            data_channels,
            input_channel,
            kernel_size=first_stage_kernel_sizes[0],
            stride=first_stage_strides[0],
            use_bn=True,
            act_func=act_func,
            ops_order="weight_bn_act",
        )
        first_block_conv = MBConvLayer(
            in_channels=input_channel,
            out_channels=first_block_width,
            kernel_size=first_stage_kernel_sizes[1],
            stride=first_stage_strides[1],
            expand_ratio=1,
            act_func=act_func,
        )
        first_block = NILMResidualBlock(first_block_conv, None)

        input_channel = first_block_width
        self.block_group_info: list[list[int]] = []
        blocks: list[nn.Module] = [first_block]
        _block_index = 1

        if self.use_frequency_features:
            input_channel = first_block_width * 2

        n_block_list = [max(self.depth_list)] * len(base_stage_width)
        for base_width, n_block, s in zip(base_stage_width, n_block_list, base_stage_strides):
            width = make_divisible(base_width * width_mult, BaseNetwork.CHANNEL_DIVISIBLE)
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = s if i == 0 else 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(input_channel, 1),
                    out_channel_list=val2list(output_channel, 1),
                    kernel_size_list=self.ks_list,
                    expand_ratio_list=self.expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                )
                mb_block = DynamicAdaptiveTransferBlock(
                    mobile_inverted_conv,
                    in_channel_list=val2list(input_channel, 1),
                    out_channel_list=val2list(output_channel, 1),
                    expand_ratio_list=self.expand_ratio_list,
                    enable_fusion=True,
                    transfer_mode="parallel",
                )
                blocks.append(mb_block)
                input_channel = output_channel

        feature_mix_layer = ConvLayer(
            input_channel,
            last_channel,
            kernel_size=1,
            use_bn=True,
            act_func=act_func,
        )
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super().__init__(first_conv, blocks, feature_mix_layer, classifier)

        if self.use_frequency_features:
            frequency_input_size = data_channels * seq_length
            self.freq_feature_extractor = FrequencyFeatureExtractor(
                input_size=frequency_input_size,
                output_size=first_block_width,
                apply_smoothing=True,
                apply_conv=True,
            )

        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        if seq2seq:
            self.seq_decoder = nn.Sequential(
                nn.Conv1d(last_channel, last_channel // 2, kernel_size=3, padding=1),
                build_activation(act_func),
                nn.Conv1d(last_channel // 2, n_classes, kernel_size=1),
            )

        self.mmd_loss = MMDLoss()
        self.coral_loss = CORALLoss()
        if "sinkhorn" in self.domain_adaptation_method:
            self.sinkhorn_loss = SinkhornLoss(
                epsilon=sinkhorn_epsilon, num_iterations=sinkhorn_iterations
            )

        self.sinkhorn_coral_weight = sinkhorn_coral_weight
        self.domain_loss_weight = domain_loss_weight
        self.domain_adaptation_mode = False

    # ---------------------------------------------------------------- transfer
    def _set_non_block_requires_grad(self, flag: bool) -> None:
        for module_name in (
            "first_conv", "feature_mix_layer", "classifier",
            "freq_feature_extractor", "seq_decoder",
        ):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = flag

    def convert_to_transfer_learning_mode(self) -> None:
        for block in self.blocks:
            if isinstance(block, DynamicAdaptiveTransferBlock):
                block.train_transfer_only()
            elif isinstance(block, NILMResidualBlock):
                for p in block.conv.parameters():
                    p.requires_grad = False
        self._set_non_block_requires_grad(False)

    def convert_to_pre_training_mode(self) -> None:
        for block in self.blocks:
            if isinstance(block, DynamicAdaptiveTransferBlock):
                block.reset_transfer_learning_mode()
            elif isinstance(block, NILMResidualBlock):
                for p in block.conv.parameters():
                    p.requires_grad = True
        self._set_non_block_requires_grad(True)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_input = x
        x = self.first_conv(x)
        time_features = self.blocks[0](x)

        if self.use_frequency_features and hasattr(self, "freq_feature_extractor"):
            freq_features = self.freq_feature_extractor(original_input)
            _batch, _c, height, seq_length = time_features.shape
            freq_spatial = freq_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, height, seq_length)
            x = torch.cat([time_features, freq_spatial], dim=1)
        else:
            x = time_features

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            for idx in block_idx[:depth]:
                x = self.blocks[idx](x)

        x = self.feature_mix_layer(x)

        if self.seq2seq:
            x = x.squeeze(2)
            output = self.seq_decoder(x)
            return output.transpose(1, 2)

        pooled = x.mean(dim=3).squeeze(2)
        return self.classifier(pooled)

    @property
    def module_str(self) -> str:
        parts = [self.first_conv.module_str, self.blocks[0].module_str]
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            for idx in block_idx[:depth]:
                parts.append(self.blocks[idx].module_str)
        parts.append(self.feature_mix_layer.module_str)
        parts.append(self.classifier.module_str)
        return "\n".join(parts)

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return self.module_str

    @property
    def config(self) -> dict:
        return {
            "name": NILMSupernet.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "feature_mix_layer": None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config: dict) -> "NILMSupernet":  # pragma: no cover
        raise NotImplementedError(
            "NILMSupernet stores state-dependent configuration; rebuild the "
            "supernet with constructor arguments and then call load_state_dict."
        )

    @property
    def grouped_block_index(self):
        return self.block_group_info

    def load_state_dict(self, state_dict, **kwargs):  # type: ignore[override]
        """Rewrites legacy key aliases so older checkpoints load cleanly."""
        model_dict = self.state_dict()
        for key in state_dict:
            new_key = key.replace(".mobile_inverted_conv.", ".conv.") if ".mobile_inverted_conv." in key else key
            if new_key in model_dict:
                pass
            elif ".bn.bn." in new_key:
                new_key = new_key.replace(".bn.bn.", ".bn.")
            elif ".conv.conv.weight" in new_key:
                new_key = new_key.replace(".conv.conv.weight", ".conv.weight")
            elif ".linear.linear." in new_key:
                new_key = new_key.replace(".linear.linear.", ".linear.")
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise KeyError(new_key)
            assert new_key in model_dict, new_key
            model_dict[new_key] = state_dict[key]
        super().load_state_dict(model_dict)

    # ------------------------------------------------------ subnet management
    def set_max_net(self) -> None:
        self.set_active_subnet(
            ks=max(self.ks_list),
            e=max(self.expand_ratio_list),
            d=max(self.depth_list),
        )

    def set_active_subnet(self, ks=None, e=None, d=None, **_kwargs) -> None:
        ks_list = val2list(ks, len(self.blocks) - 1)
        expand_list = val2list(e, len(self.blocks) - 1)
        depth_list = val2list(d, len(self.block_group_info))

        for block, k, er in zip(self.blocks[1:], ks_list, expand_list):
            if k is not None:
                block.conv.active_kernel_size = k
            if er is not None:
                block.conv.active_expand_ratio = er

        for i, depth in enumerate(depth_list):
            if depth is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), depth)

    def set_constraint(self, include_list, constraint_type: str = "depth") -> None:
        if constraint_type == "depth":
            self._depth_include_list = list(include_list)
        elif constraint_type == "expand_ratio":
            self._expand_include_list = list(include_list)
        elif constraint_type == "kernel_size":
            self._ks_include_list = list(include_list)
        else:
            raise NotImplementedError(f"unknown constraint: {constraint_type}")

    def clear_constraint(self) -> None:
        self._depth_include_list = None
        self._expand_include_list = None
        self._ks_include_list = None

    def sample_active_subnet(self) -> dict:
        """Uniformly sample a subnet configuration from the constrained space."""
        ks_candidates = getattr(self, "_ks_include_list", None) or self.ks_list
        expand_candidates = getattr(self, "_expand_include_list", None) or self.expand_ratio_list
        depth_candidates = getattr(self, "_depth_include_list", None) or self.depth_list

        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]

        ks_setting = [random.choice(cand) for cand in ks_candidates]
        expand_setting = [random.choice(cand) for cand in expand_candidates]
        depth_setting = [random.choice(cand) for cand in depth_candidates]

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)
        return {"ks": ks_setting, "e": expand_setting, "d": depth_setting}

    def get_active_subnet(self, preserve_weight: bool = True) -> NILMBackbone:
        first_conv = copy.deepcopy(self.first_conv)
        blocks = [copy.deepcopy(self.blocks[0])]
        feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
        classifier = copy.deepcopy(self.classifier)

        input_channel = blocks[0].conv.out_channels
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            for idx in block_idx[:depth]:
                block = self.blocks[idx]
                blocks.append(
                    ResidualBlock(
                        block.conv.get_active_subnet(input_channel, preserve_weight),
                        copy.deepcopy(block.shortcut),
                    )
                )
                input_channel = blocks[-1].conv.out_channels

        subnet = NILMBackbone(first_conv, blocks, feature_mix_layer, classifier)
        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def get_active_net_config(self) -> dict:
        block_config_list = [self.blocks[0].config]
        input_channel = block_config_list[0]["conv"]["out_channels"]
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            for idx in block_idx[:depth]:
                block = self.blocks[idx]
                block_config_list.append(
                    {
                        "name": ResidualBlock.__name__,
                        "conv": block.conv.get_active_subnet_config(input_channel),
                        "shortcut": block.shortcut.config if block.shortcut is not None else None,
                    }
                )
                try:
                    input_channel = block.conv.active_out_channel
                except AttributeError:
                    input_channel = block.conv.out_channels

        return {
            "name": NILMBackbone.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": block_config_list,
            "feature_mix_layer": self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    # ------------------------------------------------------------------ width
    def re_organize_middle_weights(self, expand_ratio_stage: int = 0) -> None:
        for block in self.blocks[1:]:
            block.conv.re_organize_middle_weights(expand_ratio_stage)

    # --------------------------------------------------------- domain adapt
    def get_layer_features(self, x: torch.Tensor, layer_indices=None) -> list[torch.Tensor]:
        """Collect per-layer pooled features for multi-layer domain alignment.

        The paper aligns layers ``l_1=2`` through ``l_2=4`` (Eq. 26).
        ``layer_indices=None`` returns the full stack for flexibility.
        """
        features: list[torch.Tensor] = []
        original_input = x

        x = self.first_conv(x)
        if layer_indices is None or 0 in layer_indices:
            features.append(x.mean(dim=3).squeeze(2))

        time_features = self.blocks[0](x)
        if self.use_frequency_features and hasattr(self, "freq_feature_extractor"):
            freq_features = self.freq_feature_extractor(original_input)
            _b, _c, height, seq_length = time_features.shape
            freq_spatial = freq_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, height, seq_length)
            x = torch.cat([time_features, freq_spatial], dim=1)
        else:
            x = time_features

        if layer_indices is None or 1 in layer_indices:
            features.append(x.mean(dim=3).squeeze(2))

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            for idx in block_idx[:depth]:
                x = self.blocks[idx](x)
                if layer_indices is None or idx in layer_indices:
                    features.append(x.mean(dim=3).squeeze(2))

        x = self.feature_mix_layer(x)
        pooled = x.mean(dim=3).squeeze(2)
        if layer_indices is None or -1 in layer_indices:
            features.append(pooled)
        return features

    def compute_domain_loss(
        self, source_features: list[torch.Tensor], target_features: list[torch.Tensor]
    ) -> torch.Tensor:
        if not source_features or len(source_features) != len(target_features):
            raise ValueError("source/target feature lists must be non-empty and of equal length")

        gamma = self.sinkhorn_coral_weight
        domain_loss = source_features[0].new_zeros(())
        for sf, tf in zip(source_features, target_features):
            if self.domain_adaptation_method == "mmd_coral":
                domain_loss = domain_loss + self.mmd_loss(sf, tf) + self.coral_loss(sf, tf)
            elif self.domain_adaptation_method == "sinkhorn":
                domain_loss = domain_loss + self.sinkhorn_loss(sf, tf)
            elif self.domain_adaptation_method == "sinkhorn_coral":
                domain_loss = domain_loss + gamma * self.sinkhorn_loss(sf, tf) + (1 - gamma) * self.coral_loss(sf, tf)
            elif self.domain_adaptation_method == "sinkhorn_mmd":
                domain_loss = domain_loss + self.sinkhorn_loss(sf, tf) + self.mmd_loss(sf, tf)
            else:
                raise ValueError(f"unsupported domain adaptation method: {self.domain_adaptation_method}")
        return domain_loss / len(source_features)

    def forward_domain_adaptation(
        self, source_data: torch.Tensor, target_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = [2, 3, 4]
        source_features = self.get_layer_features(source_data, layer_indices=indices)
        target_features = self.get_layer_features(target_data, layer_indices=indices)
        domain_loss = self.compute_domain_loss(source_features, target_features)
        source_pred = self.forward(source_data)
        return source_pred, domain_loss

    def enable_domain_adaptation(self, enable: bool = True) -> None:
        self.domain_adaptation_mode = enable
        if enable:
            self.convert_to_transfer_learning_mode()
        else:
            self.convert_to_pre_training_mode()

    def set_domain_loss_weight(self, weight: float) -> None:
        self.domain_loss_weight = weight

    def set_domain_adaptation_method(self, method: str) -> None:
        valid = {"mmd_coral", "sinkhorn", "sinkhorn_coral", "sinkhorn_mmd"}
        if method not in valid:
            raise ValueError(f"unsupported method '{method}', expected one of {sorted(valid)}")
        self.domain_adaptation_method = method
