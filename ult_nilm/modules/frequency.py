"""Frequency-domain feature extractor used in the time-frequency encoder.

Implements the spectral branch of Eq. (11)-(17) in the paper: the input
sequence is windowed (Hann), transformed by the discrete Fourier transform,
decomposed into amplitude and phase, projected into the channel dimension,
and concatenated with the time-domain features inside ``NILMSupernet``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ult_nilm.utils.base import BaseModule


class FrequencyFeatureExtractor(BaseModule):
    def __init__(
        self,
        input_size: int,
        output_size: int = 64,
        apply_smoothing: bool = True,
        apply_conv: bool = True,
    ):
        super().__init__()
        if input_size is None or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        self.input_size = input_size
        self.output_size = output_size
        self.apply_smoothing = apply_smoothing
        self.apply_conv = apply_conv

        if self.apply_conv:
            # Low-frequency refinement: process (real, imag) jointly.
            self.freq_conv = nn.Conv1d(2, 2, kernel_size=3, padding=1, bias=False)
        else:
            self.freq_conv = None

        # rFFT of a real input of length L yields L//2 + 1 complex bins,
        # which we split into amplitude + phase before flattening.
        fft_output_size = (self.input_size // 2 + 1) * 2
        self.projection = nn.Linear(fft_output_size, self.output_size)

    @property
    def module_str(self) -> str:
        return f"FrequencyFeatureExtractor({self.input_size}->{self.output_size})"

    @property
    def config(self) -> dict:
        return {
            "name": FrequencyFeatureExtractor.__name__,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "apply_smoothing": self.apply_smoothing,
            "apply_conv": self.apply_conv,
        }

    @staticmethod
    def build_from_config(config: dict) -> "FrequencyFeatureExtractor":
        return FrequencyFeatureExtractor(**config)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.shape[-1]
        window = torch.hann_window(seq_length, device=x.device)
        if x.dim() == 3:
            window = window.view(1, 1, -1).expand_as(x)
        elif x.dim() == 4:
            window = window.view(1, 1, 1, -1).expand_as(x)
        else:
            raise ValueError(f"unsupported input rank: {x.dim()}")
        return x * window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, 1, -1)

        if self.apply_smoothing:
            x_flat = self._smooth(x_flat)

        x_rfft = torch.fft.rfft(x_flat, dim=-1)
        real = x_rfft.real
        imag = x_rfft.imag

        if self.apply_conv and self.freq_conv is not None:
            freq_bins = real.shape[-1]
            low_freq_bins = freq_bins // 4
            if low_freq_bins > 0:
                low_complex = torch.stack(
                    [real[:, :, :low_freq_bins], imag[:, :, :low_freq_bins]],
                    dim=1,
                ).squeeze(2)
                refined = self.freq_conv(low_complex)
                real[:, :, :low_freq_bins] = refined[:, 0:1, :]
                imag[:, :, :low_freq_bins] = refined[:, 1:2, :]

        amplitude = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)

        features = torch.cat([amplitude, phase], dim=-1).view(batch_size, -1)
        return self.projection(features)
