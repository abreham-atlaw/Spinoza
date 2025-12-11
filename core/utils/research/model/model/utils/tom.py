import torch
import torch.nn as nn


class TransitionOnlyModel(nn.Module):

	def __init__(self, model: nn.Module, extra_len: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = model
		self.extra_len = extra_len

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		if self.extra_len > 0:
			X[..., X.shape[-1]-self.extra_len:] = 0.0
		out = self.model(X)
		out[:, -1] = 0
		return out
