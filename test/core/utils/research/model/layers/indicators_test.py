import unittest

import torch
import numpy as np
import matplotlib.pyplot as plt

from core.utils.research.model.layers import Indicators, PadOverlayCombiner


class IndicatorsTest(unittest.TestCase):

	def setUp(self):
		self.indicators = Indicators(
			mma=[32, 64],
			delta=[1, 2, 3],
			ksf=[(0.1, 0.1), (0.2, 0.2)],
			rsi=[32, 64],
			so=[32, 64],
			combiner=PadOverlayCombiner()
		)
		self.x = torch.from_numpy(np.load("/home/abrehamatlaw/Downloads/1758545263.358221.npy").astype(np.float32))[:, :-124]

	def test_call(self):
		y = self.indicators(self.x)
		self.assertEqual(y.shape, (self.x.shape[0], self.indicators.indicators_len, self.x.shape[1]))

		plt.figure()
		plt.title("X")
		plt.plot(self.x[0])

		for i in range(self.indicators.indicators_len):
			plt.figure()
			plt.title(f"y[{i}]")
			plt.plot(y[0, i])

		plt.show()