import os
import unittest

from core import Config
from core.di import ServiceProvider
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.losses import ProximalMaskedLoss
from core.utils.research.utils.analysis.session_analyzer import SessionAnalyzer
from lib.utils.fileio import load_json


class SessionAnalyzerTest(unittest.TestCase):

	def setUp(self):
		Config.AGENT_USE_MULTI_INSTRUMENT_MODEL = True
		Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND = load_json(os.path.join(Config.RES_DIR, "bounds/15.json"))
		self.session_analyzer = SessionAnalyzer(
			session_path=os.path.join(Config.BASE_DIR, "temp/session_dumps/05"),
			instruments=[
				("AUD", "USD"),
				("EUR", "USD")
			],
			smoothing_algorithms=[
				MovingAverage(32),
			],
			plt_y_grid_count=10,
			model_key="it-98",
			# aggregate_alpha=0.98/3
		)

	def test_plot_sequence(self):
		self.session_analyzer.plot_sequence(
			checkpoints=[(2, 0.64), 6],
			instrument=("AUD", "USD"),
			channels=("c", 'l', 'h')
		)
		# self.session_analyzer.plot_sequence(checkpoints=[2, 6], instrument=("USD", "ZAR"))

	def test_plot_timestep_sequence(self):
		self.session_analyzer.plot_timestep_sequence(i=0, instrument=("AUD", "USD"))
		self.session_analyzer.plot_timestep_sequence(i=3, instrument=("USD", "ZAR"))

	def test_evaluate_model(self):
		loss = self.session_analyzer.evaluate_loss(ProximalMaskedLoss(
			n=len(DataPrepUtils.apply_bound_epsilon(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND))
		))
		print("Loss:", loss)

	def test_plot_node(self):
		self.session_analyzer.plot_node(idx=0, depth=5, path=[])

	def test_plot_timestep_output(self):
		for i in range(1):
			self.session_analyzer.plot_timestep_output(
				i,
				h=0,
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
			path=[],
			instrument=("AUD", "USD"),
		)

	def test_plot_prediction_sequence(self):
		self.session_analyzer.plot_prediction_sequence(
			instrument=("AUD", "USD"),
			channels=(0, 1, 2),
		)

	def test_plot_trades(self):
		self.session_analyzer.plot_trades(
			channels=("c", 'l', 'h'),
		)