import os.path
import unittest

from core import Config
from core.di.legacy.swarm_agent_utils_provider import SwarmAgentUtilsProvider
from core.utils.swarm.session_setup import SwarmWorkerSetupManager


class SwarmWorkerSetupManagerTest(unittest.TestCase):

	def setUp(self):
		Config.CORE_MODEL_CONFIG.path = ""
		Config.AGENT_MODEL_TEMPERATURE = 0.0
		Config.AGENT_MODEL_AGGREGATION_ALPHA = None
		self.manager = SwarmAgentUtilsProvider.provide_worker_setup_manager()

	def test_setup(self):
		self.manager.setup()

		self.assertEqual(os.path.basename(Config.CORE_MODEL_CONFIG.path), "abrehamalemu-spinoza-training-cnn-1-it-89-tot.0.zip")
		self.assertEqual(Config.AGENT_MODEL_TEMPERATURE, 1.0)
		self.assertEqual(Config.AGENT_MODEL_AGGREGATION_ALPHA, 49.5)
