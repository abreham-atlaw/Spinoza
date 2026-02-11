import json
import unittest
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from core import Config
from core.utils.research.data.prepare import SimulationSimulator3, SimulationSimulator5
from core.utils.research.data.prepare.bound_optimizer import BoundGenerator


class BoundOptimizerTest(unittest.TestCase):

	def setUp(self):
		self.generator = BoundGenerator(
			start=0.001,
			end=20,
			csv_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-50k.csv",
			threshold=0.005,
			granularity=30,
			dataprep_class=SimulationSimulator5,
			dataprep_kwargs={
				"x_columns": ("c", 'l', 'h', 'o'),
				"y_columns": ('c', 'l', 'h', 'o'),
			}
		)

	def test_functionality(self):

		N = 12
		EXPORT_PATH = f"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/bounds/{datetime.now().timestamp()}.json"

		bounds = self.generator.generate(N, plot=True)

		with open(EXPORT_PATH, "w") as f:
			json.dump(bounds, f)
			print(f"Exported to: {EXPORT_PATH}")

		self.assertEqual(len(bounds), N)

	def test_get_weights(self):
		EXPORT_PATH = f"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/bounds/weights/{datetime.now().timestamp()}.json"

		with open("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/res/bounds/01.json", "r") as f:
			bounds = json.load(f)

		weights = self.generator.get_weights(bounds)

		with open(EXPORT_PATH, "w") as f:
			json.dump(weights, f)
			print(f"Exported to: {EXPORT_PATH}")

		plt.figure()
		plt.scatter(bounds + [bounds[-1] + Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON], weights)

		self.generator.plot_bounds(bounds)

		self.assertEqual(len(weights), len(bounds)+1)
