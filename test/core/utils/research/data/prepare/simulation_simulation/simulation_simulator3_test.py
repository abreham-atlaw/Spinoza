import os
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core import Config
from core.Config import BASE_DIR
from core.utils.research.data.prepare import SimulationSimulator3
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.splitting import SequentialSplitter


class SimulationSimulator2Test(unittest.TestCase):

	def setUp(self):
		df = pd.read_csv(os.path.join(BASE_DIR, "temp/Data/AUD-USD.2-day.csv"))
		self.output_path = os.path.join(BASE_DIR, "temp/Data/simulation_simulator_data/05")
		self.simulator = SimulationSimulator3(
			df=df,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			seq_len=128,
			extra_len=124,
			batch_size=10,
			output_path=self.output_path,
			granularity=5,
			smoothing_algorithm=MovingAverage(64),
			order_gran=True,
			trim_extra_gran=True,
			trim_incomplete_batch=True,
			splitter=SequentialSplitter(
				test_size=0.2
			),
			transformations=[
			]
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
			for idx, i in enumerate(np.argsort(np.mean(X[:, 0], axis=1))[:4]):
				plt.subplot(2, 2, idx + 1)
				for j in range(4):
					plt.plot(X[i, j, :-124], label=["c", "o", "l", "h"][j])
				plt.legend()
			plt.show()
