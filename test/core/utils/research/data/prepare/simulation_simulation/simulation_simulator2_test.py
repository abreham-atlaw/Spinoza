import os
import unittest
from datetime import timedelta, datetime

import pandas as pd
import numpy as np

from core import Config
from core.Config import BASE_DIR
from core.di import ServiceProvider
from core.utils.research.data.prepare import SimulationSimulator2
from core.utils.research.data.prepare.augmentation import VerticalShiftTransformation, GaussianNoiseTransformation, \
	TimeStretchTransformation, VerticalStretchTransformation
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage, KalmanFilter, Lass
from core.utils.research.data.prepare.splitting import SequentialSplitter
from lib.utils.torch_utils.model_handler import ModelHandler


class SimulationSimulator2Test(unittest.TestCase):

	def test_functionality(self):
		df = pd.read_csv(os.path.join(BASE_DIR, "temp/Data/AUD-USD.2-day.csv"))

		simulator = SimulationSimulator2(
			df=df,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			seq_len=128,
			extra_len=124,
			batch_size=10,
			output_path=os.path.join(BASE_DIR, "temp/Data/simulation_simulator_data/04"),
			granularity=1,
			smoothing_algorithm=ServiceProvider.provide_lass(),
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


