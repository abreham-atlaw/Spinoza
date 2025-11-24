import os.path
import unittest

import numpy as np
import pandas as pd

from core import Config
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from lib.utils.logger import Logger


class DataPrepUtilsTest(unittest.TestCase):

	def test_condense_granularity(self):
		df = pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/All-All.csv"))
		df = df[(df["base_currency"] == "AUD") & (df["quote_currency"] == "USD")]

		G = 5

		df_g = DataPrepUtils.condense_granularity(df, G)

		self.assertEqual(df_g.shape[0], df.shape[0] // G)
		self.assertEqual(df_g["l"].iloc[0], np.min(df["l"].iloc[0: G]))

		display_cols = ["time", "l", "h", "o", "c"]
		Logger.info("DF:\n", df[display_cols].head())
		Logger.success("Condensed DF:\n", df_g[display_cols].head())
