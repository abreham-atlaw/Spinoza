import typing

from torch import nn as nn

from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.resnet.residual_block import ResidualBlock


class ResNetBlock(CNNBlock):

	def _build_conv_layers(
			self,
			channels: typing.List[int],
			kernel_sizes: typing.List[int],
			stride: typing.List[int],
	) -> typing.List[nn.Module]:
		return [
			ResidualBlock(
				in_channels=channels[i],
				out_channels=channels[i + 1],
				kernel_size=kernel_sizes[i],
				init_fn=None,
				norm=False
			)
			for i in range(len(channels) - 1)
		]
