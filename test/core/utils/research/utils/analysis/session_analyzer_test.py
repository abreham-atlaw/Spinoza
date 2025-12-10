import os
import unittest

from core import Config
from core.di import ServiceProvider
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.losses import ProximalMaskedLoss
from core.utils.research.utils.analysis.session_analyzer import SessionAnalyzer


class SessionAnalyzerTest(unittest.TestCase):

	def setUp(self):
		self.session_analyzer = SessionAnalyzer(
			session_path=os.path.join(Config.BASE_DIR, "temp/session_dumps/00"),
			instruments=[
				("AUD", "USD"),
				("USD", "ZAR")
			],
			smoothing_algorithms=[
				MovingAverage(32),
			],
			plt_y_grid_count=10,
			model_key="176"
		)

	def test_plot_sequence(self):
		self.session_analyzer.plot_sequence(checkpoints=[2, 6], instrument=("AUD", "USD"))
		self.session_analyzer.plot_sequence(checkpoints=[2, 6], instrument=("USD", "ZAR"))

	def test_plot_timestep_sequence(self):
		self.session_analyzer.plot_timestep_sequence(i=3, instrument=("AUD", "USD"))
		self.session_analyzer.plot_timestep_sequence(i=3, instrument=("USD", "ZAR"))

	def test_evaluate_model(self):
		loss = self.session_analyzer.evaluate_loss(ProximalMaskedLoss(
			n=len(DataPrepUtils.apply_bound_epsilon(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND))
		))
		print("Loss:", loss)

	def test_plot_node(self):
		self.session_analyzer.plot_node(idx=0, depth=5, path=[])

	def test_plot_timestep_output(self):
		for i in range(5):
			self.session_analyzer.plot_timestep_output(
				i,
				h=1.0,
				max_depth=5,
				loss=ProximalMaskedLoss(
					n=len(DataPrepUtils.apply_bound_epsilon(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND)),
					softmax=False,
					collapsed=False
				)
			)

	def test_plot_node_prediction(self):
		self.session_analyzer.plot_node_prediction(
			0,
			path=[1, 0, 1, 0, 1, 0],
			instrument=("AUD", "USD"),
		)
