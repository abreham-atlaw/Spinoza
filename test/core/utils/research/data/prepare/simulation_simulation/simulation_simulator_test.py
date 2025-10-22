import os
import unittest
from datetime import timedelta, datetime

import pandas as pd
import numpy as np

from core import Config
from core.Config import BASE_DIR
from core.utils.research.data.prepare import SimulationSimulator
from core.utils.research.data.prepare.augmentation import VerticalShiftTransformation, GaussianNoiseTransformation, \
	TimeStretchTransformation, VerticalStretchTransformation
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage, KalmanFilter
from core.utils.research.data.prepare.splitting import SequentialSplitter


class SimulationSimulatorTest(unittest.TestCase):

	def test_functionality(self):

		df = pd.read_csv(os.path.join(BASE_DIR, "temp/Data/All-All.1-month.csv"))

		simulator = SimulationSimulator(
			df=df,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			seq_len=1033,
			extra_len=124,
			batch_size=10,
			output_path=os.path.join(BASE_DIR, "temp/Data/simulation_simulator_data/03"),
			granularity=5,
			smoothing_algorithm=MovingAverage(64),
			order_gran=True,
			trim_extra_gran=True,
			trim_incomplete_batch=True,
			splitter=SequentialSplitter(
				test_size=0.2
			),
			transformations=[
			]
		)

		simulator.start()


