import unittest

from core import Config
from core.utils.research.utils.analysis.plot_rsa import PlotRSAnalyzer


class PlotRSATest(unittest.TestCase):

	def test_functionality(self):
		Config.RunnerStatsLossesBranches.default = Config.RunnerStatsLossesBranches.it_72_1
		analyzer = PlotRSAnalyzer(
			branches=[Config.RunnerStatsBranches.it_75_8],
			# sessions_len=11,
			session_take_profits=[1.05, 1.1, 1.2],
			session_stop_losses=[0.9, 0.8, 0.7]
		)
		analyzer.start()
