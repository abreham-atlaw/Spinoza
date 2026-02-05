import unittest

import torch
from matplotlib import pyplot as plt

from core.utils.research.model.layers import AnchoredReturnsLayer


class AnchoredReturnsLayerTest(unittest.TestCase):

	def test_single_instrument(self):
		layer = AnchoredReturnsLayer(
			anchored_channels=[0, 1, 2, 3],
			anchor_channels=0,
			log=True
		)

		x = torch.rand((5, 4, 10))
		y = layer(x)

		self.assertEqual(y.shape, torch.Size((5, 4, 9)))



		for i in range(x.shape[0]):
			print(f"X[{i}]", x)
			print(f"y[{i}]", y)

			plt.figure()

			plt.subplot(1, 2, 1)
			plt.plot(x[i].transpose(1, 0))

			plt.subplot(1, 2, 2)
			plt.plot(y[i].transpose(1, 0))

		plt.show()
