import os
import unittest
from datetime import datetime

from core.Config import BASE_DIR
from core.utils.research.data.prepare import SimulationTimesGenerator


class SimulationTimesGeneratorTest(unittest.TestCase):

	def setUp(self):
		self.generator = SimulationTimesGenerator(
			random_mode=False
		)

	def test_generate(self):
		self.generator.generate(
			start_time=datetime(year=2025, month=1, day=28),
			end_time=datetime(year=2025, month=8, day=1),
			count=50,
			export_path=os.path.join(BASE_DIR, "res/times/times-50-it-4.json")
		)
