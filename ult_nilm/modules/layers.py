"""Static layer primitives used to assemble the NILM elastic supernet.

These are the plain (non-elastic) building blocks. The elastic counterparts
live in ``ult_nilm.networks.dynamic_layers`` and share the same configuration
interface so that a sampled sub-architecture can be serialised and rebuilt.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from ult_nilm.utils.base import BaseModule
from ult_nilm.utils.common import build_activation, get_same_padding, min_divisible_value


def set_layer_from_config(layer_config: dict | None):
    if layer_config is None:
        return None
    name2layer = {
        ConvLayer.__name__: ConvLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        ZeroLayer.__name__: ZeroLayer,
        MBConvLayer.__name__: MBConvLayer,
        ResidualBlock.__name__: ResidualBlock,
        NILMResidualBlock.__name__: NILMResidualBlock,
    }
    layer_name = layer_config.pop("name")
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class _Base2DLayer(BaseModule):
    """Shared scaffolding for Conv/Identity layers with optional BN + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
        act_func: str | None = "relu",
        dropout_rate: float = 0.0,
        ops_order: str = "weight_bn_act",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        modules: dict[str, object] = {}
        if self.use_bn:
            feat = in_channels if self.bn_before_weight else out_channels
            modules["bn"] = nn.BatchNorm2d(feat)
        else:
            modules["bn"] = None
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act" and self.use_bn)
        modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True) if dropout_rate > 0 else None
        modules["weight"] = self.weight_op()

        for op in self.ops_list:
            if modules[op] is None:
                continue
            if op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self) -> list[str]:
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self) -> bool:
        for op in self.ops_list:
            if op == "bn":
                return True
            if op == "weight":
                return False
        raise ValueError(f"invalid ops_order: {self.ops_order}")

    def weight_op(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def config(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }


class ConvLayer(_Base2DLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        act_func: str | None = "relu",
        dropout_rate: float = 0.0,
        ops_order: str = "weight_bn_act",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        super().__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding = (padding[0] * self.dilation, padding[1] * self.dilation)
        return OrderedDict(
            {
                "conv": nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    dilation=self.dilation,
                    groups=min_divisible_value(self.in_channels, self.groups),
                    bias=self.bias,
                )
            }
        )

    @property
    def module_str(self) -> str:
        k = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
        name = f"{k[0]}x{k[1]}_Conv" if self.groups == 1 else f"{k[0]}x{k[1]}_GroupConv"
        return f"{name}_O{self.out_channels}_{str(self.act_func).upper()}{'_BN' if self.use_bn else ''}"

    @property
    def config(self) -> dict:
        return {
            "name": ConvLayer.__name__,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            **super().config,
        }

    @staticmethod
    def build_from_config(config: dict) -> "ConvLayer":
        return ConvLayer(**config)


class IdentityLayer(_Base2DLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = False,
        act_func: str | None = None,
        dropout_rate: float = 0.0,
        ops_order: str = "weight_bn_act",
    ):
        super().__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        return None

    @property
    def module_str(self) -> str:
        return "Identity"

    @property
    def config(self) -> dict:
        return {"name": IdentityLayer.__name__, **super().config}

    @staticmethod
    def build_from_config(config: dict) -> "IdentityLayer":
        return IdentityLayer(**config)


class LinearLayer(BaseModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_bn: bool = False,
        act_func: str | None = None,
        dropout_rate: float = 0.0,
        ops_order: str = "weight_bn_act",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        modules: dict[str, object] = {}
        if self.use_bn:
            feat = in_features if self.bn_before_weight else out_features
            modules["bn"] = nn.BatchNorm1d(feat)
        else:
            modules["bn"] = None
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")
        modules["dropout"] = nn.Dropout(self.dropout_rate, inplace=True) if dropout_rate > 0 else None
        modules["weight"] = {"linear": nn.Linear(in_features, out_features, bias)}

        for op in self.ops_list:
            if modules[op] is None:
                continue
            if op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self) -> list[str]:
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self) -> bool:
        for op in self.ops_list:
            if op == "bn":
                return True
            if op == "weight":
                return False
        raise ValueError(f"invalid ops_order: {self.ops_order}")

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self) -> str:
        return f"{self.in_features}x{self.out_features}_Linear"

    @property
    def config(self) -> dict:
        return {
            "name": LinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config: dict) -> "LinearLayer":
        return LinearLayer(**config)


class ZeroLayer(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise ValueError("ZeroLayer should be short-circuited by its container")

    @property
    def module_str(self) -> str:
        return "Zero"

    @property
    def config(self) -> dict:
        return {"name": ZeroLayer.__name__}

    @staticmethod
    def build_from_config(config: dict) -> "ZeroLayer":
        return ZeroLayer()


class MBConvLayer(BaseModule):
    """Mobile inverted bottleneck: 1x1 expand -> depthwise -> 1x1 project."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        mid_channels: int | None = None,
        act_func: str = "relu6",
        groups: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.groups = groups

        feature_dim = mid_channels if mid_channels is not None else round(in_channels * expand_ratio)
        pad = get_same_padding(kernel_size)
        groups_eff = feature_dim if groups is None else min_divisible_value(feature_dim, groups)

        if feature_dim != in_channels:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(in_channels, feature_dim, 1, 1, 0, bias=False)),
                        ("bn", nn.BatchNorm2d(feature_dim)),
                        ("act", build_activation(act_func, inplace=True)),
                    ]
                )
            )
        else:
            self.inverted_bottleneck = None

        self.depth_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            feature_dim,
                            feature_dim,
                            kernel_size,
                            stride,
                            pad,
                            groups=groups_eff,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(feature_dim)),
                    ("act", build_activation(act_func, inplace=True)),
                ]
            )
        )
        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                    ("bn", nn.BatchNorm2d(out_channels)),
                ]
            )
        )

    def forward(self, x):
        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self) -> str:
        expand = self.expand_ratio if self.mid_channels is None else self.mid_channels // self.in_channels
        return (
            f"{self.kernel_size}x{self.kernel_size}_MBConv{expand}_"
            f"{self.act_func.upper()}_O{self.out_channels}"
        )

    @property
    def config(self) -> dict:
        return {
            "name": MBConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "groups": self.groups,
        }

    @staticmethod
    def build_from_config(config: dict) -> "MBConvLayer":
        return MBConvLayer(**config)


class ResidualBlock(BaseModule):
    def __init__(self, conv, shortcut):
        super().__init__()
        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.conv is None or isinstance(self.conv, ZeroLayer):
            return x
        if self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            return self.conv(x)
        return self.conv(x) + self.shortcut(x)

    @property
    def module_str(self) -> str:
        return (
            f"({self.conv.module_str if self.conv is not None else None}, "
            f"{self.shortcut.module_str if self.shortcut is not None else None})"
        )

    @property
    def config(self) -> dict:
        return {
            "name": ResidualBlock.__name__,
            "conv": self.conv.config if self.conv is not None else None,
            "shortcut": self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config: dict) -> "ResidualBlock":
        conv = set_layer_from_config(config.get("conv"))
        shortcut = set_layer_from_config(config["shortcut"])
        return ResidualBlock(conv, shortcut)

    @property
    def mobile_inverted_conv(self):
        return self.conv


class NILMResidualBlock(ResidualBlock):
    """Residual block with edge-aware shortcut gating.

    At an on/off appliance transition, ``|F(x) - x|`` spikes; the shortcut is
    attenuated at those positions so the edge information flows through the
    convolution path instead of being washed out by the skip connection.
    """

    def __init__(self, conv, shortcut, edge_threshold: float = 0.5):
        super().__init__(conv, shortcut)
        self.edge_threshold = edge_threshold

    def forward(self, x):
        conv_out = self.conv(x)
        if self.shortcut is None:
            return conv_out
        edge_mask = (torch.abs(conv_out - x) > self.edge_threshold).float()
        shortcut_out = self.shortcut(x) * (1.0 - edge_mask)
        return conv_out + shortcut_out

    @property
    def config(self) -> dict:
        cfg = super().config
        cfg["name"] = NILMResidualBlock.__name__
        cfg["edge_threshold"] = self.edge_threshold
        return cfg

    @staticmethod
    def build_from_config(config: dict) -> "NILMResidualBlock":
        edge_threshold = config.pop("edge_threshold", 0.5)
        conv = set_layer_from_config(config.get("conv"))
        shortcut = set_layer_from_config(config["shortcut"])
        return NILMResidualBlock(conv, shortcut, edge_threshold)
