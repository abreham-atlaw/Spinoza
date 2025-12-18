import numpy as np
import pandas as pd

import os
import unittest

from matplotlib import pyplot as plt

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass3_preparer import Lass3Preparer
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass7_preparer import Lass7Preparer
from core.utils.research.data.prepare.splitting import SequentialSplitter
from core.utils.research.data.prepare.utils.sinusoidal_decomposer import SinusoidalDecomposer
from lib.utils.logger import Logger


class Lass7PreparerTest(unittest.TestCase):

	def setUp(self):
		self.output_path = os.path.join(Config.BASE_DIR, "temp/Data/lass/14")
		Logger.info(f"Cleaning {self.output_path}")
		os.system(f"rm -fr \"{self.output_path}\"")
		self.preparer = Lass7Preparer(
			decomposer=SinusoidalDecomposer(min_block_size=512, plot_progress=False),
			block_size=32,
			granularity=2,
			batch_size=64,
			output_path=self.output_path,
			order_gran=True,
			df=pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD-10k.csv")),
			splitter=SequentialSplitter(test_size=0.2),
			left_align=False,
			decoder_samples=8

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
			for idx, i in enumerate(np.argsort(np.sum(X[:, 1,] == 0, axis=1))[:4]):
				plt.subplot(2, 2, idx + 1)
				plt.plot(X[i, 0], label="X-Encoder")
				plt.plot(X[i, 1][X[i, 1] > 0], label="X-Decoder")
				plt.scatter([np.sum(X[i, 1] > 0)], [y[i]], label="Y", c="red")
			plt.show()

