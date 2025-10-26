import torch
from torch import nn

from core.utils.research.model.model.savable import SpinozaModule


class EncoderDropout(SpinozaModule):

	def __init__(self, dropout: float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.args = {
			"dropout": dropout
		}
		self.dropout = nn.Dropout(dropout)

	def call(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
		return self.dropout(x_encoder)

	def export_config(self) -> dict:
		return self.args
