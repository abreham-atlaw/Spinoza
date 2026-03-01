import os.path
import unittest

from core import Config
from core.di import ResearchProvider
from core.utils.research.data.collect.sim_setup.rs_setup_manager import RSSetupManager


class RSSetupManagerTest(unittest.TestCase):

	def setUp(self):

		Config.RunnerStatsBranches.default = Config.RunnerStatsBranches.it_98_6
		Config.RunnerStatsLossesBranches.default = Config.RunnerStatsLossesBranches.it_101_0
		Config.OANDA_SIM_MODEL_IN_PATH = "/Apps/RTrader/maploss/it-98/"

		Config.OANDA_TRADING_ACCOUNT_ID = ""
		Config.OANDA_TRADING_URL = "http://127.0.0.1:8000/api"

		Config.UPDATE_SAVE_PATH = os.path.join(Config.BASE_DIR, "temp/session_dumps/06/outs")
		Config.SESSION_MODEL_EVALUATION = True

		self.setup_manager: RSSetupManager = ResearchProvider.provide_rs_setup_manager()

	def test_setup(self):
		stat = self.setup_manager.setup()

		self.assertNotEqual(Config.OANDA_TRADING_ACCOUNT_ID, "")
		self.assertEqual(stat.model_name, os.path.basename(Config.CORE_MODEL_CONFIG.path))
		print(stat)

	def test_setup_and_finish(self):
		stat = self.setup_manager.setup()

		PL = 1.2
		self.setup_manager.finish(
			stat=stat,
			pl=PL,
		)

		print(stat)
