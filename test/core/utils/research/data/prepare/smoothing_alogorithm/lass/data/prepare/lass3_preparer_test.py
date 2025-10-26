import pandas as pd

import os
import unittest

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass3_preparer import Lass3Preparer
from core.utils.research.data.prepare.splitting import SequentialSplitter


class Lass3PreparerTest(unittest.TestCase):

	def setUp(self):
		self.preparer = Lass3Preparer(
			sa=MovingAverage(window_size=32),
			shift=16,
			block_size=32,
			granularity=30,
			batch_size=64,
			output_path=os.path.join(Config.BASE_DIR, "temp/Data/lass/7"),
			order_gran=True,
			df=pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD-10k.csv")),
			splitter=SequentialSplitter(test_size=0.2),
			process_batch_size=4096,
			left_align=False,
			decoder_samples=8
		)

	def test_functionality(self):
		self.preparer.start()
