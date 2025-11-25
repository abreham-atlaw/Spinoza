import os
import unittest
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from core.Config import BASE_DIR
from core.utils.research.data.prepare.augmentation import Transformation


class TransformationAbstractTest(unittest.TestCase, ABC):

	@abstractmethod
	def _init_transformation(self) -> Transformation:
		pass

	def setUp(self):
		self.transformation = self._init_transformation()
		self.x = np.load(os.path.join(BASE_DIR, "temp/Data/simulation_simulator_data/00/train/X/1758248772.176.npy"))[:, :-124]
		self.x_3d = np.stack([
			np.sin(
				np.linspace(0,  1 * np.pi * ((i+1)**3), 1024),
			).reshape((-1, 128))
			for i in range(4)
		], axis=1)

	def test_transform(self):
		y = self.transformation.transform(self.x)
		SAMPLES = 5
		for i in range(SAMPLES):
			plt.figure()
			plt.plot(self.x[i])
			plt.plot(y[i])
		plt.show()

	def test_3d_input(self):
		y = self.transformation(self.x_3d)
		SAMPLES = 5
		for i in range(SAMPLES):
			plt.figure()
			for j in range(self.x_3d.shape[1]):
				plt.subplot(self.x_3d.shape[1], 1, j+1)
				plt.grid()
				plt.plot(self.x_3d[i, j], label="x")
				plt.plot(y[i, j], label="y")
				plt.legend()
		plt.show()
