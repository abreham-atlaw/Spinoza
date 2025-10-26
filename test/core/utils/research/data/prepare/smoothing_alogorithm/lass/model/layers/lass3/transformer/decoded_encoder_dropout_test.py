import unittest

import torch
from matplotlib import pyplot as plt

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers.lass3.transformer import \
	DecodedEncoderDropout


class DecodedEncoderDropoutTest(unittest.TestCase):

	def setUp(self):
		self.layer = DecodedEncoderDropout(0.5)
		self.layer.train()

	def test_functionality(self):
		x_encoder = torch.rand(5, 10)
		x_decoder = torch.zeros_like(x_encoder)
		x_decoder[torch.arange(5).reshape(-1, 1) >= (8-torch.arange(10))] = 0.5

		y = self.layer(x_encoder, x_decoder)

		for i in range(x_encoder.shape[0]):
			plt.figure()
			plt.plot(x_encoder[i].numpy(), label="x_encoder")
			plt.plot(x_decoder[i][x_decoder[i] > 0].numpy(), label="x_decoder")
			plt.plot(y[i].numpy(), label="y")

		plt.legend()
		plt.show()