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

	def test_transform(self):
		y = self.transformation.transform(self.x)
		SAMPLES = 5
		for i in range(SAMPLES):
			plt.figure()
			plt.plot(self.x[i])
			plt.plot(y[i])
		plt.show()
