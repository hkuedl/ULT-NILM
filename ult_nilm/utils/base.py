"""Base module/network classes providing a uniform configuration interface."""

from __future__ import annotations

import math

import torch.nn as nn


class BaseModule(nn.Module):
    """Base class that enforces a serialisable `config` contract on every layer."""

    def forward(self, x):  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def module_str(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def config(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def build_from_config(config: dict) -> "BaseModule":  # pragma: no cover - abstract
        raise NotImplementedError


class BaseNetwork(BaseModule):
    """Base class for top-level networks.

    Provides utility methods shared across the elastic supernet and any static
    subnet extracted from it.
    """

    CHANNEL_DIVISIBLE = 8

    def zero_last_gamma(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def grouped_block_index(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def set_bn_param(self, momentum: float, eps: float) -> None:
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = momentum
                m.eps = eps

    def get_bn_param(self) -> dict | None:
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                return {"momentum": m.momentum, "eps": m.eps}
        return None

    def get_parameters(self, keys=None, mode: str = "include"):
        if keys is None:
            for _name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
            return
        if mode not in {"include", "exclude"}:
            raise ValueError(f"unsupported mode: {mode}")
        for name, param in self.named_parameters():
            matched = any(key in name for key in keys)
            if mode == "include" and matched and param.requires_grad:
                yield param
            elif mode == "exclude" and not matched and param.requires_grad:
                yield param

    def weight_parameters(self):
        return self.get_parameters()


def get_net_device(net):
    """Return the device of the first trainable parameter in a module."""
    return next(net.parameters()).device


def count_parameters(net) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def init_models(net, model_init: str = "he_fout") -> None:
    """Kaiming-style weight initialisation for Conv/Linear/BN layers."""
    if isinstance(net, list):
        for sub_net in net:
            init_models(sub_net, model_init)
        return
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if model_init == "he_fout":
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            elif model_init == "he_fin":
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            else:
                raise NotImplementedError(f"unsupported init: {model_init}")
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.zero_()
