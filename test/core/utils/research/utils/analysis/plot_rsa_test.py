import unittest

from core import Config
from core.utils.research.utils.analysis.plot_rsa import PlotRSAnalyzer


class PlotRSATest(unittest.TestCase):

	def test_functionality(self):
		Config.RunnerStatsLossesBranches.default = Config.RunnerStatsLossesBranches.it_76_0
		analyzer = PlotRSAnalyzer(
			branches=[Config.RunnerStatsBranches.it_76_6],
			sessions_len=11
		)
		analyzer.start()
