import unittest

import torch

from core.utils.research.model.model.titan import TitanContextEmbeddingBlock


class TitanContextEmbeddingBlockTest(unittest.TestCase):

	def setUp(self):
		self.instrument_positions = (3, 5)
		self.instrument_vocab = 2
		self.block = TitanContextEmbeddingBlock(
			instrument_positions=self.instrument_positions,
			instruments_vocab=self.instrument_vocab,
			embedding_size=10,
		)

	def test_call(self):
		x = torch.randn(5, 10)

		for idx, ins in zip(self.instrument_positions, range(self.instrument_vocab)):
			x[:, idx] = torch.randint(0, self.instrument_vocab, (x.shape[0],))

		print(f"X: {x}")

		y = self.block(x)

		print(y)


