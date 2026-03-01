import typing

import numpy as np
import torch
import torch.nn as nn

from core.utils.research.losses import SpinozaLoss


class SentimentLoss(SpinozaLoss):

	def __init__(
			self,
			bounds: typing.List[float],
			*args,
			multi_channel: bool = True,
			channels_weight: typing.List[float] = None,
			d: float = 0,
			bound_neutral: float = 1,
			softmax: bool = True,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.multi_channel = multi_channel
		self.d = d
		if isinstance(bounds, np.ndarray):
			bounds = torch.from_numpy(bounds)
		self.bounds = bounds
		if not isinstance(bounds, torch.Tensor):
			raise TypeError(f"Bounds should be of type torch.Tensor or np.ndarray. Got {type(bounds)}")
		self.bounds -= bound_neutral
		self.softmax = nn.Softmax(dim=-1) if softmax else nn.Identity()
		self.channels_weight = torch.Tensor(channels_weight) if channels_weight is not None else None

	def __get_channel_weight(self, i: int) -> float:
		return self.channels_weight[i] if self.channels_weight is not None else 1

	def get_sentiment(self, y: torch.Tensor) -> torch.Tensor:
		s = torch.sum(self.bounds * y, dim=-1)
		s += self.d * torch.sign(s)
		return s

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

		if self.multi_channel and y.ndim == 3:
			loss = torch.stack([
					self._call(y_hat[:, i], y[:, i])
					for i in range(y.shape[1])
				], dim=1)
			if self.channels_weight is not None:
				loss = torch.sum((loss*self.channels_weight)/torch.sum(self.channels_weight), dim=1)
			else:
				loss = torch.mean(loss, dim=1)
			return loss

		y_hat = self.softmax(y_hat)

		y_hat_sentiment = self.get_sentiment(y_hat)
		y_sentiment = self.get_sentiment(y)
		return torch.abs(y_hat_sentiment - y_sentiment)
