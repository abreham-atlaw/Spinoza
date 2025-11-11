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
		self.output_path = os.path.join(Config.BASE_DIR, "temp/Data/lass/13")
		Logger.info(f"Cleaning {self.output_path}")
		os.system(f"rm -fr \"{self.output_path}\"")
		c_x = 100
		f = 0.05 + (1e-4 * (np.arange(c_x)**4))
		a = (1/f)**0.77

		self.preparer = Lass6Preparer(
			output_path=self.output_path,

			seq_size=int(2e3),
			block_size=128,
			batch_size=1024,
			splitter=SequentialSplitter(test_size=0.2),

			transformations=[
				VerticalShiftTransformation(shift=1.5),
			],

			c_x=c_x,
			c_y=11,
			noise=0,
			noise_p=15,
			f=f,
			a=a,
			target_mean=0.7,
			target_std=7e-3
		)

	def test_functionality(self):
		self.preparer.start()

		container_path = os.path.join(self.output_path, "train")

		X_FILES, Y_FILES = [
			[
				os.path.join(container_path, axis, filename)
				for filename in sorted(os.listdir(os.path.join(os.path.join(container_path, axis))))
			]
			for axis in ["X", "y"]
		]

		for f in np.random.randint(0, len(X_FILES), 10):
			plt.figure(figsize=(20, 10))
			X, y = [np.load(files[f]) for files in [X_FILES, Y_FILES]]
			for idx, i in enumerate(np.argsort(np.mean(X[:, 0], axis=1))[:4]):
				plt.subplot(2, 2, idx + 1)
				plt.plot(X[i, 0], label="X-Encoder")
				plt.plot(X[i, 1][X[i, 1] > 0], label="X-Decoder")
				plt.scatter([np.sum(X[i, 1] > 0)], [y[i]], label="Y", c="red")
			plt.show()

