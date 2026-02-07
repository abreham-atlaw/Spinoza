import unittest

from core import Config
from core.utils.research.utils.analysis.plot_rsa import PlotRSAnalyzer


class PlotRSATest(unittest.TestCase):

	def test_functionality(self):
		Config.RunnerStatsLossesBranches.default = Config.RunnerStatsLossesBranches.it_72_1
		analyzer = PlotRSAnalyzer(
			branches=[Config.RunnerStatsBranches.it_73_7],
			sessions_len=11
		)
		analyzer.start()
