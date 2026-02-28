import numpy as np
import pandas as pd

import os
import unittest

from matplotlib import pyplot as plt

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass10_preparer import Lass10Preparer
from core.utils.research.data.prepare.splitting import SequentialSplitter
from core.utils.research.data.prepare.utils.sinusoidal_decomposer import SinusoidalDecomposer
from lib.utils.logger import Logger


class Lass10PreparerTest(unittest.TestCase):

	def setUp(self):
		self.output_path = os.path.join(Config.BASE_DIR, "temp/Data/lass/15")
		Logger.info(f"Cleaning {self.output_path}")
		os.system(f"rm -fr \"{self.output_path}\"")
		self.x_columns = ("c", "l", "h", "o")
		self.y_columns = ("c",)
		self.batch_size = 64
		self.block_size = 32
		self.preparer = Lass10Preparer(
			decomposer=SinusoidalDecomposer(min_block_size=1024, block_layers=3, blocks_rate=2.5, plot_progress=False),
			block_size=self.block_size,
			granularity=2,
			batch_size=self.batch_size,
			output_path=self.output_path,
			order_gran=True,
			df=pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/Al-All.10k.csv")),
			splitter=SequentialSplitter(test_size=0.2),
			left_align=False,
			vertical_align=True,
			x_columns=self.x_columns,
			y_columns=self.y_columns
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
			self.assertEqual(X.shape, (self.batch_size, len(self.x_columns)*2, self.block_size))
			self.assertEqual(y.shape, (self.batch_size, len(self.y_columns)*2, self.block_size))
			for idx, i in enumerate(np.random.randint(0, X.shape[0], 2)):
				plt.subplot(2, 2, idx + 1)
				plt.plot(X[i].transpose(1, 0), label=[f"X-{c}" for c in self.x_columns*2])
				plt.plot(y[i].transpose(1, 0), label=[f"Y-{c}" for c in self.y_columns*2])
			plt.show()

