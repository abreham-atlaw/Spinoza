import torch

from .overlay_indicator import OverlayIndicator


class StochasticOscillator(OverlayIndicator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = 1e-7

    def _on_time_point(self, inputs: torch.Tensor) -> torch.Tensor:
        highest = torch.max(inputs, dim=2)[0]
        lowest = torch.min(inputs, dim=2)[0]
        close = inputs[:, :, -1]
        return (close - lowest) / (highest - lowest + self.epsilon)
