import os
import unittest
from datetime import datetime

import pandas as pd

from core.Config import BASE_DIR
from core.utils.research.data.prepare.datasource_preparer import MITDatasourcePreparer


class MITDatasourcePreparerTest(unittest.TestCase):

	def setUp(self):
		self.preparer = MITDatasourcePreparer(
			export_path=os.path.join(BASE_DIR, "temp/Data/All-All.1-month.csv"),
		)

	def test_prepare_multiple(self):
		self.preparer.prepare_multiple(
			paths=[
				os.path.join(BASE_DIR, path)
				for path in [
					"temp/Data/USD-ZAR.csv",
					"temp/Data/AUD-USD.csv"
				]
			],
			time_range=(
				pd.to_datetime("2022-01-01 00:00:00+00:00"),
				pd.to_datetime("2022-02-01 00:00:00+00:00")
			)
		)
