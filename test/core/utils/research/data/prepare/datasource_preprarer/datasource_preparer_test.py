import unittest
from datetime import datetime, timezone

from core.utils.research.data.prepare.datasource_preparer import DatasourcePreparer


class DatasourcePreparerTest(unittest.TestCase):

	def setUp(self):
		self.preparer = DatasourcePreparer(
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prep"
		)
		self.DF_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-2k.csv"
		self.time_range = datetime(year=2003, month=1, day=1, tzinfo=timezone.utc), datetime(year=2004, month=1, day=1, tzinfo=timezone.utc)

	def test_prepare(self):
		self.preparer.prepare(path=self.DF_PATH, time_range=self.time_range)
