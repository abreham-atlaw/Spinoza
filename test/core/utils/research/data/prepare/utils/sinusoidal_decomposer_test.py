import unittest

import pandas as pd

import os

from core import Config
from core.utils.research.data.prepare.utils.sinusoidal_decomposer import SinusoidalDecomposer


class SinusoidalDecomposerTest(unittest.TestCase):

	def setUp(self):
		self.decomposer = SinusoidalDecomposer(
			layer_indifference_threshold=0.05,
			correction_steps=2,
			use_correction=True
		)

	def test_optimize(self):
		df = pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD-10k.csv"))
		x = df["c"].to_numpy()[::2]
		y = self.decomposer.decompose(x)
