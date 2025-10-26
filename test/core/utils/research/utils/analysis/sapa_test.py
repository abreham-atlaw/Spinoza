import unittest

from matplotlib import pyplot as plt

import os

from core import Config
from core.di import ServiceProvider
from core.utils.research.data.prepare.smoothing_algorithm import KalmanFilter, MovingAverage
from core.utils.research.utils.analysis.sapa import SmoothingAlgorithmProfitabilityAnalyzer


class SmoothingAlgorithmProfitabilityAnalyzerTest(unittest.TestCase):

	def setUp(self):
		self.analyzer = SmoothingAlgorithmProfitabilityAnalyzer(
			df_path=os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD.csv"),
			view_size=128,
			tp_threshold=5,
			action_lag_size=(18, 20),
			plot_show=False,
			samples=4,
			plot=True,
			granularity=30
		)

		self.sas = [
			MovingAverage(64),
			ServiceProvider.provide_lass()
		]

	def test_analyze(self):
		for sa in self.sas:
			self.analyzer.analyze(sa)
		plt.show()
