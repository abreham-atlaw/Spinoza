import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import unittest

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass3_preparer import Lass3Preparer
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass4_preparer import Lass4Preparer
from core.utils.research.data.prepare.splitting import SequentialSplitter
from lib.utils.logger import Logger


class Lass3PreparerTest(unittest.TestCase):

	def setUp(self):
		self.export_path = os.path.join(Config.BASE_DIR, "temp/Data/lass/6")
		self.preparer = Lass4Preparer(
			sa=MovingAverage(window_size=64),
			shift=32,
			block_size=128,
			granularity=1,
			batch_size=64,
			output_path=self.export_path,
			order_gran=True,
			df=pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD-10k.csv")),
			splitter=SequentialSplitter(test_size=0.2),
			process_batch_size=4096,
			left_align=False
		)

	def test_functionality(self):
		self.preparer.start()

		SAMPLES = 5

		X_dir, y_dir = [
			os.path.join(self.export_path, "train", axis)
			for axis in ["X", "y"]
		]

		filenames = os.listdir(X_dir)

		filename = random.choice(filenames)
		Logger.info(f"Using filename: {filename}")
		X, y = [
			np.load(os.path.join(path, filename))
			for path in [X_dir, y_dir]
		]

		idxs = np.random.randint(0, X.shape[0], SAMPLES)

		for i in idxs:
			plt.figure()
			plt.title(f"i={i}\ny={y[i]}")
			plt.plot(X[i, 0], label="X-Encoder")
			plt.plot(X[i, 1][X[i, 1] > 0], label="X-Decoder")
			plt.scatter([np.sum(X[i, 1] > 0)], [y[i] + X[i, 1, -1]], label="Y", c="red")
		plt.legend()
		plt.show()
