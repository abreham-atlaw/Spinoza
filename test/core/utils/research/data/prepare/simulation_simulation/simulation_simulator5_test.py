import os
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core import Config
from core.Config import BASE_DIR
from core.utils.research.data.prepare.simulation_simulator import SimulationSimulator5
from core.utils.research.data.prepare.smoothing_algorithm import  IdentitySA
from core.utils.research.data.prepare.splitting import SequentialSplitter
from lib.utils.fileio import load_json
from lib.utils.logger import Logger


class SimulationSimulator5Test(unittest.TestCase):

	def setUp(self):
		df = pd.read_csv(os.path.join(BASE_DIR, "temp/Data/data_sources/mit-1-test.10k.csv"))
		self.output_path = os.path.join(BASE_DIR, "temp/Data/simulation_simulator_data/12")

		Logger.warning(f"Cleaning output path: {self.output_path}...")
		os.system(f"rm -fr \"{self.output_path}\"")

		self.channels = ("c", "l", "h", "o")
		self.extra_len = 12

		self.simulator = SimulationSimulator5(
			df=df,
			bounds=load_json(os.path.join(Config.BASE_DIR, "res/bounds/15.json")),
			seq_len=128,
			extra_len=self.extra_len,
			batch_size=128,
			output_path=self.output_path,
			granularity=2,
			smoothing_algorithm=IdentitySA(),
			smoothed_columns=(),
			x_columns=self.channels,
			y_columns=self.channels,
			order_gran=True,
			trim_extra_gran=True,
			trim_incomplete_batch=True,
			splitter=SequentialSplitter(
				test_size=0.2
			),
			transformations=[
			],
		)

	def test_functionality(self):

		self.simulator.start()

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
			if self.extra_len > 0:
				X = X[..., :self.extra_len]
			for idx, i in enumerate(np.argsort(np.mean(X[:, 0], axis=1))[:4]):
				plt.subplot(4, 2, 2*idx+1)
				for j in range(X.shape[1]):
					plt.plot(X[i, j], label=self.channels[j % len(self.channels)])

				plt.subplot(4, 2, 2*idx+2)
				for j in range(X.shape[1]):
					plt.plot(y[i, j], label=self.channels[j % len(self.channels)])
				plt.legend()
			plt.show()
