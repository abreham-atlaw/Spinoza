import unittest

import torch

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers.lass3.transformer import DecoderPadEmbedding


class DecoderPadEmbeddingTest(unittest.TestCase):

	def setUp(self):
		self.layer = DecoderPadEmbedding()

	def test_functionality(self):
		x = torch.zeros((5, 10))
		x[torch.arange(5).reshape(-1, 1) >= (9-torch.arange(10))] = 0.5
		y = self.layer(x.clone())
		print(x)
		print(y)
