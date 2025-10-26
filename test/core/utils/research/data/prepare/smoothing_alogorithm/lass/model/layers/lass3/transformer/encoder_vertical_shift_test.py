import unittest

import torch
from matplotlib import pyplot as plt

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers.lass3.transformer import EncoderVerticalShift


class EncoderChannelDropoutTest(unittest.TestCase):

	def setUp(self):
		self.layer = EncoderVerticalShift(1.0)

	def test_functionality(self):
		x_encoder = torch.rand(5, 128)
		x_decoder = torch.zeros_like(x_encoder)
		y = self.layer(x_encoder.clone(), x_decoder)

		for i in range(x_encoder.shape[0]):
			plt.figure()
			plt.plot(x_encoder[i].numpy(), label="x_encoder")
			plt.plot(x_decoder[i][x_decoder[i] > 0].numpy(), label="x_decoder")
			plt.plot(y[i].numpy(), label="y")
		plt.legend()
		plt.show()

		self.assertEqual(x_encoder.shape, y.shape)