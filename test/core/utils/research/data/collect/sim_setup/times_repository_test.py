import unittest

import os

from core import Config
from core.di import ResearchProvider
from core.utils.research.data.collect.runner_stats2 import RunnerStats2
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
from core.utils.research.data.collect.sim_setup.times_repository import JsonTimesRepository


class TimesRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.repo = JsonTimesRepository(path=os.path.join(Config.RES_DIR, "times/times-50.json"))

	def test_allocate(self):
		rs_repo: RunnerStatsRepository = ResearchProvider.provide_runner_stats_repository(Config.RunnerStatsBranches.it_75_6)
		stat = rs_repo.retrieve_all()[0]

		print(stat.simulated_timestamps)

		time = self.repo.allocate(
			stat
		)

		print(time)

