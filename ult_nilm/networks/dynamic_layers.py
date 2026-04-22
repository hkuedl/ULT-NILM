"""Elastic layer blocks composed from the dynamic operators.

These layers are the workhorses that the elastic supernet stacks into elastic
units. Each block retains full-size tensors internally and exposes two
interfaces:

* ``forward`` uses the currently-selected sub-configuration.
* ``get_active_subnet`` materialises a detached static layer from
  ``ult_nilm.modules.layers`` so that sampled subnets can be exported and
  run without the elastic overhead.
"""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from ult_nilm.modules.layers import (
    ConvLayer,
    LinearLayer,
    MBConvLayer,
    set_layer_from_config,
)
from ult_nilm.networks.dynamic_ops import (
    DynamicBatchNorm2d,
    DynamicConv2d,
    DynamicLinear,
    DynamicSeparableConv2d,
)
from ult_nilm.utils.base import BaseModule, BaseNetwork, get_net_device
from ult_nilm.utils.common import build_activation, make_divisible, val2list


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def copy_bn(target_bn, src_bn):
    feature_dim = target_bn.num_features
    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if isinstance(src_bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


class DynamicLinearLayer(BaseModule):
    def __init__(
        self,
        in_features_list,
        out_features: int,
        bias: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_features_list = in_features_list
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate, inplace=True) if dropout_rate > 0 else None
        self.linear = DynamicLinear(max(in_features_list), out_features, bias)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self) -> str:
        return f"DyLinear({max(self.in_features_list)}, {self.out_features})"

    @property
    def config(self) -> dict:
        return {
            "name": DynamicLinearLayer.__name__,
            "in_features_list": self.in_features_list,
            "out_features": self.out_features,
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config: dict) -> "DynamicLinearLayer":
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features: int, preserve_weight: bool = True) -> LinearLayer:
        sub_layer = LinearLayer(in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer
        sub_layer.linear.weight.data.copy_(
            self.linear.get_active_weight(self.out_features, in_features).data
        )
        if self.bias:
            sub_layer.linear.bias.data.copy_(self.linear.get_active_bias(self.out_features).data)
        return sub_layer

    def get_active_subnet_config(self, in_features: int) -> dict:
        return {
            "name": LinearLayer.__name__,
            "in_features": in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
        }


class DynamicConvLayer(BaseModule):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        use_bn: bool = True,
        act_func: str = "relu6",
    ):
        super().__init__()
        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicConv2d(
            max_in_channels=max(in_channel_list),
            max_out_channels=max(out_channel_list),
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        if use_bn:
            self.bn = DynamicBatchNorm2d(max(out_channel_list))
        self.act = build_activation(act_func)

        self.active_out_channel = max(out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    @property
    def module_str(self) -> str:
        return f"DyConv(O{self.active_out_channel}, K{self.kernel_size}, S{self.stride})"

    @property
    def config(self) -> dict:
        return {
            "name": DynamicConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config: dict) -> "DynamicConvLayer":
        return DynamicConvLayer(**config)

    @property
    def in_channels(self) -> int:
        return max(self.in_channel_list)

    @property
    def out_channels(self) -> int:
        return max(self.out_channel_list)

    def get_active_subnet(self, in_channel: int, preserve_weight: bool = True) -> ConvLayer:
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer
        sub_layer.conv.weight.data.copy_(
            self.conv.get_active_filter(self.active_out_channel, in_channel).data
        )
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)
        return sub_layer

    def get_active_subnet_config(self, in_channel: int) -> dict:
        return {
            "name": ConvLayer.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }


class DynamicMBConvLayer(BaseModule):
    """Mobile inverted bottleneck with elastic kernel/width/expansion."""

    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size_list=3,
        expand_ratio_list=6,
        stride: int = 1,
        act_func: str = "relu6",
    ):
        super().__init__()
        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size_list = val2list(kernel_size_list)
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.stride = stride
        self.act_func = act_func

        max_middle_channel = make_divisible(
            round(max(self.in_channel_list) * max(self.expand_ratio_list)),
            BaseNetwork.CHANNEL_DIVISIBLE,
        )

        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", DynamicConv2d(max(self.in_channel_list), max_middle_channel)),
                        ("bn", DynamicBatchNorm2d(max_middle_channel)),
                        ("act", build_activation(act_func)),
                    ]
                )
            )

        self.depth_conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv", DynamicSeparableConv2d(max_middle_channel, self.kernel_size_list, stride)),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(act_func)),
                ]
            )
        )
        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", DynamicConv2d(max_middle_channel, max(self.out_channel_list))),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                ]
            )
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)
        middle_channel = make_divisible(
            round(in_channel * self.active_expand_ratio),
            BaseNetwork.CHANNEL_DIVISIBLE,
        )

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = middle_channel
            x = self.inverted_bottleneck(x)

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self) -> str:
        return f"(O{self.active_out_channel}, E{self.active_expand_ratio:.1f}, K{self.active_kernel_size})"

    @property
    def config(self) -> dict:
        return {
            "name": DynamicMBConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "expand_ratio_list": self.expand_ratio_list,
            "stride": self.stride,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config: dict) -> "DynamicMBConvLayer":
        return DynamicMBConvLayer(**config)

    @property
    def in_channels(self) -> int:
        return max(self.in_channel_list)

    @property
    def out_channels(self) -> int:
        return max(self.out_channel_list)

    def active_middle_channel(self, in_channel: int) -> int:
        return make_divisible(
            round(in_channel * self.active_expand_ratio), BaseNetwork.CHANNEL_DIVISIBLE
        )

    def get_active_subnet(self, in_channel: int, preserve_weight: bool = True) -> MBConvLayer:
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        middle_channel = self.active_middle_channel(in_channel)
        if self.inverted_bottleneck is not None and sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.get_active_filter(middle_channel, in_channel).data
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)
        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)
        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.get_active_filter(self.active_out_channel, middle_channel).data
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)
        return sub_layer

    def get_active_subnet_config(self, in_channel: int) -> dict:
        return {
            "name": MBConvLayer.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.active_kernel_size,
            "stride": self.stride,
            "expand_ratio": self.active_expand_ratio,
            "mid_channels": self.active_middle_channel(in_channel),
            "act_func": self.act_func,
        }

    def re_organize_middle_weights(self, expand_ratio_stage: int = 0) -> None:
        """Channel sort so the most important filters live at the low index.

        Implements the importance-based reordering used during progressive
        shrinking: smaller subnets always select the prefix of channels and
        keeping that prefix channel-important improves their stand-alone
        performance.
        """
        importance = torch.sum(torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3))
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(
                    round(max(self.in_channel_list) * expand), BaseNetwork.CHANNEL_DIVISIBLE
                )
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = -len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left

        _sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )
        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )


class DynamicAdaptiveTransferBlock(BaseModule):
    """Wraps a frozen convolution with a trainable residual branch.

    This block is the structural backbone of the lightweight cross-domain
    transfer path in the paper (Section III.C, final paragraph): during
    on-device adaptation, ``conv`` and ``shortcut`` are frozen while the
    compact ``transfer_branch`` and (optional) ``fusion`` gate learn the
    domain-specific residual.
    """

    def __init__(
        self,
        conv,
        in_channel_list,
        out_channel_list,
        expand_ratio_list=6,
        reduction: int = 16,
        transfer_mode: str = "parallel",
        enable_fusion: bool = True,
    ):
        super().__init__()
        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.reduction = reduction
        self.transfer_mode = transfer_mode
        self.enable_fusion = enable_fusion
        self.transfer_learning_mode = False
        self.conv = conv

        channels = make_divisible(
            round(max(self.in_channel_list) * max(self.expand_ratio_list)),
            BaseNetwork.CHANNEL_DIVISIBLE,
        )
        out_channels = max(self.out_channel_list)
        self.active_out_channel = out_channels

        self.transfer_branch = nn.Sequential(
            OrderedDict(
                [
                    ("conv_dw", DynamicSeparableConv2d(channels, [3], stride=1)),
                    ("bn1", DynamicBatchNorm2d(channels)),
                    ("act1", nn.ReLU(inplace=True)),
                    ("conv_pw", DynamicConv2d(channels, out_channels)),
                    ("bn2", DynamicBatchNorm2d(out_channels)),
                ]
            )
        )

        if self.enable_fusion:
            self.fusion = nn.Sequential(
                OrderedDict(
                    [
                        ("pool", nn.AdaptiveAvgPool2d(1)),
                        ("conv", DynamicConv2d(out_channels, 2)),
                        ("bn", DynamicBatchNorm2d(2)),
                        ("act", nn.Sigmoid()),
                    ]
                )
            )
            inner = getattr(self.fusion.conv, "conv", None)
            if inner is not None and hasattr(inner, "weight"):
                nn.init.normal_(inner.weight, std=0.01)
                if inner.bias is not None:
                    nn.init.zeros_(inner.bias)
        else:
            self.fusion = None

        self.shortcut = None

    def train_transfer_only(self) -> None:
        self.transfer_learning_mode = True
        for p in self.conv.parameters():
            p.requires_grad = False
        if self.shortcut is not None:
            for p in self.shortcut.parameters():
                p.requires_grad = False
        for p in self.transfer_branch.parameters():
            p.requires_grad = True
        if self.enable_fusion:
            for p in self.fusion.parameters():
                p.requires_grad = True

    def reset_transfer_learning_mode(self) -> None:
        self.transfer_learning_mode = False
        for p in self.conv.parameters():
            p.requires_grad = True
        if self.shortcut is not None:
            for p in self.shortcut.parameters():
                p.requires_grad = True

    def forward(self, x):
        main_feat = self.conv(x)
        if self.shortcut is not None:
            main_feat = main_feat + self.shortcut(x)

        if not self.transfer_learning_mode and not self.enable_fusion:
            return main_feat

        if self.transfer_mode == "parallel":
            transfer_feat = self.transfer_branch(x)
        else:
            transfer_feat = self.transfer_branch(main_feat)

        if self.enable_fusion:
            fusion_weights = self.fusion(main_feat)
            main_weight = fusion_weights[:, 0:1, :, :]
            transfer_weight = fusion_weights[:, 1:2, :, :]
            return main_feat * main_weight + transfer_feat * transfer_weight
        return main_feat + transfer_feat

    @property
    def module_str(self) -> str:
        return f"AT({self.conv.module_str})"

    @property
    def config(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "conv": self.conv.config if self.conv is not None else None,
            "shortcut": self.shortcut.config if self.shortcut is not None else None,
            "transfer_mode": self.transfer_mode,
            "enable_fusion": self.enable_fusion,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "expand_ratio_list": self.expand_ratio_list,
            "reduction": self.reduction,
        }

    @staticmethod
    def build_from_config(config: dict) -> "DynamicAdaptiveTransferBlock":
        return DynamicAdaptiveTransferBlock(**config)

    @property
    def in_channels(self) -> int:
        return max(self.in_channel_list)

    @property
    def out_channels(self) -> int:
        return max(self.out_channel_list)

    def get_active_subnet(self, in_channel: int, preserve_weight: bool = True):
        raise NotImplementedError(
            "DynamicAdaptiveTransferBlock is extracted via NILMSupernet.get_active_subnet, "
            "which operates on the inner dynamic conv only."
        )

    def get_active_subnet_config(self, in_channel: int) -> dict:
        return self.conv.get_active_subnet_config(in_channel)
