import torch

from .pml3 import ProximalMaskedLoss3

class ProximalMaskedLoss4(ProximalMaskedLoss3):

	def __init__(self, *args, s=0.15, **kwargs):
		self.s = s
		super().__init__(*args, **kwargs)

	def __standardize(self, x: torch.Tensor) -> torch.Tensor:
		return x / torch.max(x, dim=-1, keepdim=True).values

	def _generate_mask(self) -> torch.Tensor:
		x = super()._generate_mask()
		s = torch.unsqueeze(
			self._generate_curve(a=0, w=self.s),
			dim=0
		)
		x = x * s
		x = self.__standardize(x)
		return x
