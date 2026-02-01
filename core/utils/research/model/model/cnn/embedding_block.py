import typing

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute1D

from core.utils.research.model.layers import Indicators, DynamicLayerNorm, IndicatorsSet
from core.utils.research.model.model.savable import SpinozaModule


class EmbeddingBlock(SpinozaModule):

	def __init__(
			self,
			indicators: typing.Union[Indicators, IndicatorsSet] = None,
			positional_encoding: bool = False,
			norm_positional_encoding: bool = False,
			input_norm: nn.Module = None,
			input_dropout: typing.Union[float, nn.Module] = 0,
			padding: nn.Module = None,
			indicators_mask: typing.Union[typing.List[bool], nn.Module] = None,
			prep_layer: typing.Optional[nn.Module] = None
	):
		self.args = {
			"indicators": indicators,
			"positional_encoding": positional_encoding,
			"norm_positional_encoding": norm_positional_encoding,
			"input_norm": input_norm,
			"input_dropout": input_dropout,
			"padding": padding,
			"indicators_mask": indicators_mask,
			"prep_layer": prep_layer
		}
		super().__init__(auto_build=False)
		self.indicators = indicators if indicators is not None else Indicators()
		self.pos_layer = None

		self.pos_norm = DynamicLayerNorm() if norm_positional_encoding else nn.Identity()
		self.pos = self.positional_encoding if positional_encoding else nn.Identity()

		self.input_norm = input_norm if input_norm is not None else nn.Identity()

		self.input_dropout = self.__prepare_dropout_args(input_dropout)
		self.padding = padding if padding is not None else nn.Identity()

		self.indicators_mask = self.__prepare_indicators_mask_args(indicators_mask)
		self.prep_layer = prep_layer if prep_layer is not None else nn.Identity()

	@staticmethod
	def __prepare_dropout_args(input_dropout: typing.Union[float, nn.Module]):
		if isinstance(input_dropout, nn.Module):
			return input_dropout
		return nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()

	@staticmethod
	def __prepare_indicators_mask_args(indicators_mask: typing.Union[typing.List[bool], nn.Module]) -> typing.Optional[nn.Module]:
		if indicators_mask is None:
			return None

		if not isinstance(indicators_mask, nn.Module) and isinstance(indicators_mask, typing.Iterable):
			return torch.Tensor(indicators_mask)

		return indicators_mask

	def positional_encoding(self, inputs: torch.Tensor) -> torch.Tensor:
		if self.pos_layer is None:
			self.pos_layer = PositionalEncodingPermute1D(inputs.shape[1])
		inputs = self.pos_norm(inputs)
		return inputs + self.pos_layer(inputs)

	def apply_indicators(self, x: torch.Tensor) -> torch.Tensor:
		if self.indicators_mask is None:
			return self.indicators(x)

		return torch.concatenate([
			self.indicators(x[:, self.indicators_mask]),
			x[:, ~self.indicators_mask]
		], dim=1)

	def call(self, x: torch.Tensor) -> torch.Tensor:
		out = self.prep_layer(x)
		out = self.input_dropout(out)
		out = self.input_norm(out)

		out = self.apply_indicators(out)

		out = self.pos(out)
		out = self.padding(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
