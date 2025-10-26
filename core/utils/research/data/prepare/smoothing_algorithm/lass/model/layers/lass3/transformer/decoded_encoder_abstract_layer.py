from abc import abstractmethod, ABC

import torch

from core.utils.research.model.model.savable import SpinozaModule


class DecodedEncoderAbstractLayer(SpinozaModule, ABC):

	def __init__(self, left_align: bool = False):
		self.args = {
			"left_align": left_align
		}
		super().__init__()
		self.left_align = left_align

	def get_decoded_mask(self, x_decoder: torch.Tensor) -> torch.Tensor:
		mask = torch.sign(x_decoder) > 0
		if not self.left_align:
			mask = torch.flip(mask, dims=(-1,))
		return mask

	@abstractmethod
	def _call(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor, decoded_mask: torch.Tensor) -> torch.Tensor:
		pass

	def call(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
		decoded_mask = self.get_decoded_mask(x_decoder)
		return self._call(x_encoder, x_decoder, decoded_mask)

	def export_config(self):
		return self.args