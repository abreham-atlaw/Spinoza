import torch
from torch import nn


class Log(nn.Module):

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.log(x)
