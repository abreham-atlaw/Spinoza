import unittest

from core.di.legacy.swarm_agent_utils_provider import SwarmAgentUtilsProvider


class SwarmQueenSetupManagerTest(unittest.TestCase):

	def setUp(self):
		self.manager = SwarmAgentUtilsProvider.provide_queen_setup_manager()

	def test_setup(self):
		self.manager.setup()
		self.manager._sio.wait()