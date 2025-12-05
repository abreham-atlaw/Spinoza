import unittest

from core import Config
from core.utils.research.utils.analysis.plot_rsa import PlotRSAnalyzer


class PlotRSATest(unittest.TestCase):

	def test_functionality(self):
		Config.RunnerStatsLossesBranches.default = Config.RunnerStatsLossesBranches.it_74_0
		analyzer = PlotRSAnalyzer(
			branches=[Config.RunnerStatsBranches.it_74_6, Config.RunnerStatsBranches.it_75_6],
			sessions_len=10
		)
		analyzer.start()
