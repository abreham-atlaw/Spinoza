import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers import EncoderNoiseInjectionLayer


class EncoderNoiseInjectionLayerTest(unittest.TestCase):

	def setUp(self):
		self.layer = EncoderNoiseInjectionLayer(noise=5e-3, frequency=1.0)

	def test_functionality(self):
		x_encoder = torch.unsqueeze(
			torch.arange(20).float() / 20,
			dim=0
		)
		x_decoder = torch.zeros_like(x_encoder)
		x_decoder[:, 5:] = 0.5

		out = self.layer(x_encoder, x_decoder)

		plt.plot(x_encoder[0], label="x_encoder")
		plt.plot(x_decoder[x_decoder > 0.0], label="x_decoder")
		plt.plot(out[0], label="out")
		plt.legend()
		plt.show()

	def test_real_data(self):
		X = torch.from_numpy(
			np.load("/home/abrehamatlaw/Downloads/1754751450.572768(2).npy").astype(np.float32),
		)
		x_encoder = X[:, 0]
		x_decoder = X[:, 1]

		y = self.layer(x_encoder, x_decoder)

		SAMPLES = 5

		idxs = torch.randint(x_encoder.shape[0], (SAMPLES,))

		for i in idxs:
			plt.figure()
			plt.grid(True)
			plt.plot(x_encoder[i].numpy(), label="x_encoder")
			plt.plot(x_decoder[i][x_decoder[i] > 0].numpy(), label="x_decoder")
			plt.plot(y[i].numpy(), label="y")
			plt.legend()
			plt.title(f"i={i}")
		plt.show()
