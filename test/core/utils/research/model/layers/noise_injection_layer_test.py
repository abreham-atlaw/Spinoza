import os
import unittest

import numpy as np
import torch
import matplotlib.pyplot as plt

from core import Config
from core.utils.research.model.layers import NoiseInjectionLayer


class NoiseInjectionLayerTest(unittest.TestCase):

	def setUp(self):
		self.layer = NoiseInjectionLayer(
			noise=5e-5,
			bounds=(0.8, 1.0)
		)

	def test_call(self):
		x = torch.from_numpy(np.load(os.path.join(Config.BASE_DIR, "temp/Data/simulation_simulator_data/02/train/X/1758508564.413.npy")))[:, :-124]
		y = self.layer(x)

		for i in np.random.randint(0, x.shape[0], 5):
			plt.figure()
			plt.plot(x[i])
			plt.plot(y[i])

		plt.show()
