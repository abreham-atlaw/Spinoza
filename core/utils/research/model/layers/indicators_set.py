import typing

import torch
from torch import nn

from core.utils.research.model.layers import Indicators, OverlaysCombiner
from core.utils.research.model.model.savable import SpinozaModule


class IndicatorsSet(SpinozaModule):

	def __init__(
			self,
			channels: typing.List[typing.Tuple[int, ...]],
			indicators: typing.List[Indicators],
			combiner: SpinozaModule = None
	):
		self.args = {
			"channels": channels,
			"indicators": indicators,
			"combiner": combiner
		}
		super().__init__()
		assert len(channels) == len(indicators)
		self.indicators = list(zip(channels, indicators))
		self.combiner = combiner if combiner is not None else OverlaysCombiner()

	@property
	def indicators_len(self) -> int:
		return sum([
			len(channels)
			if isinstance(indicator, nn.Identity)
			else
			indicator.indicators_len
			for channels, indicator in self.indicators
		])

	def call(self, x: torch.Tensor) -> torch.Tensor:

		y = []

		for channels, indicator in self.indicators:
			y.extend(torch.transpose(
				indicator(x[:, channels]),
				0, 1
			))

		return self.combiner(y)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
