import pandas as pd

import os
import unittest

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass5_preparer import Lass5Preparer
from core.utils.research.data.prepare.splitting import SequentialSplitter


class Lass5PreparerTest(unittest.TestCase):

	def setUp(self):
		self.preparer = Lass5Preparer(
			sa=MovingAverage(window_size=32),
			shift=16,
			block_size=32,
			granularity=30,
			batch_size=64,
			output_path=os.path.join(Config.BASE_DIR, "temp/Data/lass/9"),
			order_gran=True,
			df=pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD-10k.csv")),
			splitter=SequentialSplitter(test_size=0.2),
			process_batch_size=4096,
		)

	def test_functionality(self):
		self.preparer.start()
