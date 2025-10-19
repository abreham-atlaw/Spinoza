import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import unittest

from core import Config
from core.utils.research.data.prepare.augmentation import VerticalShiftTransformation, VerticalStretchTransformation, \
	TimeStretchTransformation, GaussianNoiseTransformation
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass6_preparer import Lass6Preparer
from core.utils.research.data.prepare.splitting import SequentialSplitter
from lib.utils.logger import Logger


class Lass6PreparerTest(unittest.TestCase):

	def setUp(self):
		self.output_path = os.path.join(Config.BASE_DIR, "temp/Data/lass/10")
		Logger.info(f"Cleaning {self.output_path}")
		os.system(f"rm -fr \"{self.output_path}\"")
		self.preparer = Lass6Preparer(
			output_path=self.output_path,

			seq_size=int(1e3),
			block_size=128,
			batch_size=1024,
			splitter=SequentialSplitter(test_size=0.2),

			transformations=[
				VerticalShiftTransformation(shift=0.1),
				VerticalStretchTransformation(alpha=1.1),
				VerticalStretchTransformation(alpha=0.9),
				TimeStretchTransformation(),
				GaussianNoiseTransformation()
			],

			c_x=int(1e3),
			c_y=25,
			noise=0.75e-4,
			f=1.2,
			a=1.0,
			target_mean=0.7,
			target_std=5e-3
		)

	def test_functionality(self):
		self.preparer.start()

		filename = random.choice(os.listdir(os.path.join(self.output_path, "train/X")))
		X, y = [
			np.load(os.path.join(self.output_path, f"train/{axis}/{filename}"))
			for axis in ["X", "y"]
		]

		for i in np.random.randint(0, X.shape[0], 10):
			plt.figure()
			plt.plot(X[i, 0], label="X-Encoder")
			plt.plot(X[i, 1][X[i, 1] > 0], label="X-Decoder")
			plt.scatter([np.sum(X[i, 1] > 0)], [y[i]], label="Y", c="red")
			plt.legend()
		plt.show()

