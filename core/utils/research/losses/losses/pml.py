import typing

import numpy as np
import torch
from torch import nn

from core.utils.research.losses import SpinozaLoss


class ProximalMaskedLoss(SpinozaLoss):

	def __init__(
			self,
			n: int,
			*args,
			p: int = 1,
			e: float = 1,
			lr: float = 1,
			weights: typing.Optional[typing.Union[torch.Tensor, typing.List[float], np.ndarray]] = None,
			softmax=True,
			epsilon=1e-9,
			device='cpu',
			multi_channel: bool=False,
			channels_weight: typing.List[float] = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.n = n
		self.p = p
		self.e = e
		self.lr = lr
		self.activation = nn.Softmax(dim=-1) if softmax else nn.Identity()
		self.mask = self._generate_mask().to(device)
		self.epsilon = epsilon
		self.device = device

		if weights is None:
			weights = torch.ones(n)
		if isinstance(weights, list):
			weights = torch.Tensor(weights)
		if isinstance(weights, np.ndarray):
			weights = torch.from_numpy(weights)

		self.weights = weights.to(device)
		self.multi_channel = multi_channel
		self.channels_weight = channels_weight

	def __get_channel_weight(self, i: int) -> float:
		return self.channels_weight[i] if self.channels_weight is not None else 1

	def _f(self, i: int) -> torch.Tensor:
		return (1 / (torch.abs(torch.arange(self.n) - i) + 1)) ** self.p

	def _generate_mask(self) -> torch.Tensor:

		return torch.stack([
			self._f(t) for t in range(self.n)
		])

	def collapse(self, loss: torch.Tensor) -> torch.Tensor:
		return torch.mean(loss)

	def _loss(self, y_mask: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
		return (1 / (torch.sum(y_mask * y_hat, dim=1) - self.epsilon)) - 1

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

		if self.multi_channel and y.ndim == 3:
			return torch.mean(
				torch.stack([
					self._call(y_hat[:, i], y[:, i]) * self.__get_channel_weight(i)
					for i in range(y_hat.shape[1])
				],dim=1),
				dim=1
			)

		y_hat = self.activation(y_hat)
		y_hat = y_hat**self.e

		y_mask = torch.sum(
			self.mask * torch.unsqueeze(y, dim=2),
			dim=1
		)

		loss = self._loss(y_mask, y_hat)**(1/self.lr)
		w = torch.sum(self.weights * y, dim=1)
		loss = loss*w

		return loss
