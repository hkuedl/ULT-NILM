"""Dynamic convolution and normalisation primitives.

These operators hold full-size weight tensors while exposing a slicing view
over them, so that smaller sub-architectures can be sampled without separate
storage. The elastic supernet (``ult_nilm.networks.elastic.NILMSupernet``) is
built on top of these primitives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ult_nilm.utils.common import get_same_padding, sub_filter_start_end


class DynamicSeparableConv2d(nn.Module):
    """Depthwise conv with elastic kernel size.

    Smaller kernels are obtained by centre-cropping the maximum-kernel weights
    and optionally projecting through learnable transform matrices so that the
    representational capacity is preserved across kernel choices.
    """

    KERNEL_TRANSFORM_MODE = 1  # None disables the learnable kernel transform

    def __init__(self, max_in_channels: int, kernel_size_list, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv2d(
            max_in_channels,
            max_in_channels,
            max(kernel_size_list),
            stride,
            groups=max_in_channels,
            bias=False,
        )

        self._ks_set = sorted(set(kernel_size_list))
        if self.KERNEL_TRANSFORM_MODE is not None:
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = f"{ks_larger}to{ks_small}_matrix"
                self.register_parameter(param_name, Parameter(torch.eye(ks_small**2)))

        self.active_kernel_size = max(kernel_size_list)

    def get_active_filter(self, in_channel: int, kernel_size: int):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)
        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]

        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                s, e = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, s:e, s:e].contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(_input_filter, getattr(self, f"{src_ks}to{target_ks}_matrix"))
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size: int | None = None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)
        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        padding = get_same_padding(kernel_size)
        return F.conv2d(x, filters, None, self.stride, padding, self.dilation, in_channel)


class DynamicConv2d(nn.Module):
    """Elastic pointwise conv with variable output channel count."""

    def __init__(
        self,
        max_in_channels: int,
        max_out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv2d(max_in_channels, max_out_channels, kernel_size, stride=stride, bias=False)
        self.active_out_channel = max_out_channels

    def get_active_filter(self, out_channel: int, in_channel: int):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel: int | None = None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()
        padding = get_same_padding(self.kernel_size)
        return F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)


class DynamicBatchNorm2d(nn.Module):
    """BatchNorm wrapper that selects a subset of running statistics per forward."""

    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim: int):
        super().__init__()
        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim: int):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        exponential_average_factor = 0.0
        if bn.training and bn.track_running_stats:
            if bn.num_batches_tracked is not None:
                bn.num_batches_tracked += 1
                if bn.momentum is None:
                    exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                else:
                    exponential_average_factor = bn.momentum
        return F.batch_norm(
            x,
            bn.running_mean[:feature_dim],
            bn.running_var[:feature_dim],
            bn.weight[:feature_dim],
            bn.bias[:feature_dim],
            bn.training or not bn.track_running_stats,
            exponential_average_factor,
            bn.eps,
        )

    def forward(self, x):
        return self.bn_forward(x, self.bn, x.size(1))


class DynamicLinear(nn.Module):
    """Elastic fully connected layer with variable output feature count."""

    def __init__(self, max_in_features: int, max_out_features: int, bias: bool = True):
        super().__init__()
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias
        self.linear = nn.Linear(max_in_features, max_out_features, bias)
        self.active_out_features = max_out_features

    def get_active_weight(self, out_features: int, in_features: int):
        return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features: int):
        return self.linear.bias[:out_features] if self.bias else None

    def forward(self, x, out_features: int | None = None):
        if out_features is None:
            out_features = self.active_out_features
        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        return F.linear(x, weight, bias)
