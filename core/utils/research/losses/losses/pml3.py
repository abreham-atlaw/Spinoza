import typing

import numpy as np
import torch

from .pml2 import ProximalMaskedLoss2


class ProximalMaskedLoss3(ProximalMaskedLoss2):

	def __init__(
			self,
			bounds: typing.Union[np.ndarray, torch.Tensor],
			*args,
			w: float = 0.4,
			h: float = -5.0,
			b: float = 3e-2,
			c: float = 5.0,
			m: float = 2.6,
			r: int = 3,
			d: float = 5.0,
			**kwargs
	):

		if isinstance(bounds, np.ndarray):
			bounds = torch.from_numpy(bounds)

		self.bounds = bounds
		self.m = m
		self.r = r
		self.d = d

		super().__init__(
			n=bounds.shape[0],
			*args,
			w=w,
			h=h,
			b=b,
			c=c,

			**kwargs
		)

	def __v(self, x: torch.Tensor) -> torch.Tensor:
		return self.bounds[x]

	@staticmethod
	def __root(x: torch.Tensor, r: int) -> torch.Tensor:
		return torch.sign(x) * torch.abs(x) ** (1 / r)

	def _abscissa(self, x: torch.Tensor) -> torch.Tensor:
		v = self.__v(x) - 1
		return (10**self.m)*(self.__root(v, self.r)) + self.d*torch.sign(v)

	def __str__(self):
		return f"ProximalMaskedLoss3(h={self.h}, b={self.b}, c={self.c}, m={self.m}, r={self.r}, d={self.d}, e={self.e})"