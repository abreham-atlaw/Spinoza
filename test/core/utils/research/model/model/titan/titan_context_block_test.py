import unittest

import torch

from core.utils.research.model.layers import DynamicLayerNorm
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.titan import TitanContextEmbeddingBlock
from core.utils.research.model.model.titan.titan_context_block import TitanContextBlock
from core.utils.research.model.model.transformer import DecoderBlock


class TitanContextBlockTest(unittest.TestCase):

	def setUp(self):
		self.instrument_positions = [3, 6]
		self.instrument_vocab = 2
		self.block = TitanContextBlock(
			embedding_block=TitanContextEmbeddingBlock(
				instrument_positions=self.instrument_positions,
				instruments_vocab=self.instrument_vocab,
				embedding_size=10
			),
			context_ffn=LinearModel(
				layer_sizes=[128, 64]
			),
			time_series_decoder_block=DecoderBlock(
				num_heads=4,
				norm_1=DynamicLayerNorm(),
				norm_2=DynamicLayerNorm(),
			),
			time_series_ffn=LinearModel(
				layer_sizes=[128, 64, 1],
			),
			value_head=LinearModel(
				layer_sizes=[254, 128, 1]
			)
		)

	def test_call(self):
		x = torch.randn((5, 64, 10))
		for idx, ins in zip(self.instrument_positions, range(self.instrument_vocab)):
			x[..., idx] = torch.randint(0, self.instrument_vocab, (x.shape[0],1))


		time_series_latent = torch.randn((5, 64, 128))
		y = self.block(x, time_series_latent)

		print(y)
