import os
import random
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import Config
from core.di import ServiceProvider
from core.utils.research.data.prepare.smoothing_algorithm import Lass, MovingAverage
from core.utils.research.data.prepare.smoothing_algorithm.lass.executors import Lass2Executor, Lass3Executor, \
	Lass4Executor, Lass5PlainExecutor
from lib.utils.torch_utils.model_handler import ModelHandler


class LassTest(unittest.TestCase):

	def setUp(self):
		self.lass = ServiceProvider.provide_lass()
		self.df = pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD-50k.csv"))
		self.sequence = self.df["c"].to_numpy()[-int(3e4):]
		self.gran = 30

	def test_mock(self):
		x = (np.arange(1024) + 5) * ((np.random.rand(1024) * 0) + 1)
		y = self.lass.apply(x)
		self.assertIsInstance(y, np.ndarray)
		print(y)
		plt.plot(x)
		plt.plot(y)
		plt.show()

	def test_functionality(self):
		x = self.sequence[::self.gran]
		y = self.lass.apply(x)
		self.assertIsInstance(y, np.ndarray)
		plt.plot(np.arange(len(x)), x)
		plt.plot(y)
		plt.show()

	def test_batch(self):
		batch_size = 128
		x = self.sequence
		x = np.stack([
			x[i*batch_size: (i+1)*batch_size]
			for i in range(batch_size)
		])
		y = self.lass.apply(x)
		self.assertIsInstance(y, np.ndarray)
		plt.plot(np.arange(len(x)), x)
		plt.plot(y)
		plt.show()

	def test_plot_output(self):
		ma = MovingAverage(32)

		x = self.sequence[::self.gran]
		y = x
		for i in range(1):
			y = self.lass(y)
		y_ma = ma(x)

		x, y_ma = [
			arr[-y.shape[0]:]
			for arr in [x, y_ma]
		]

		view_window = 128
		samples = 6
		cols = 3
		rows = int(np.ceil(samples/cols))

		idxs = np.random.randint(0, x.shape[0]-view_window, samples)

		plt.figure()
		for i, j in enumerate(idxs):
			plt.subplot(rows, cols, i+1)
			plt.grid(True)
			for arr in [x, y, y_ma]:
				plt.plot(arr[j: j+view_window])

		plt.show()
