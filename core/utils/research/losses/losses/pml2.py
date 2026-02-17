import typing

import torch

from .pml import ProximalMaskedLoss


class ProximalMaskedLoss2(ProximalMaskedLoss):

	def __init__(self, *args, w: float = 1.0, h: float = 5.0, b: float = 1e-3, c: float = 0, **kwargs):
		self.w = w
		self.h = h
		self.b = b
		self.c = c
		super().__init__(*args, **kwargs)

	@staticmethod
	def _abscissa(x: torch.Tensor) -> torch.Tensor:
		return x

	def _generate_curve(
			self,
			a: typing.Union[torch.Tensor, float],
			w: float
	) -> torch.Tensor:
		x = torch.arange(self.n)
		return (
				(
						(torch.e ** self.c) /
						(1 + torch.exp(
							w * torch.abs(
								self._abscissa(x)-a
							) - self.h
						))
				) + self.b) * (1/(1+self.b))

	def _f(
			self,
			i: int,
	) -> torch.Tensor:
		return self._generate_curve(
			self._abscissa(i),
			w=self.w
		)
