import os.path
import unittest

import torch
import numpy as np
from matplotlib import pyplot as plt

from core import Config
from core.utils.research.model.model.utils import AggregateModel, WrappedModel, TemperatureScalingModel
from lib.utils.torch_utils.model_handler import ModelHandler


class AggregateModelTest(unittest.TestCase):


	def setUp(self):
		self.raw_model = WrappedModel(
			TemperatureScalingModel(
				temperature=100.0,
				model=ModelHandler.load(os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-training-cnn-1-it-69-tot.zip")).eval()
			),
			seq_len=128
		)
		self.model = AggregateModel(
			model=self.raw_model,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			a=0.95/5
		).eval()

		self.X = torch.from_numpy(np.load(os.path.join(Config.BASE_DIR, "temp/Data/prepared/9/X/1765596927.065502.npy")).astype(np.float32))

	def test_aggregate(self):
		with torch.no_grad():
			y = self.raw_model(self.X)[..., :-1]
			y_hat = self.model.aggregate(y)

		for i in np.random.randint(0, self.X.shape[0], 5):
			plt.figure()
			plt.plot(y[i], label="Raw")
			plt.plot(y_hat[i], label="Aggregated")
			plt.legend()

		plt.show()

	def test_forward(self):

		with torch.no_grad():
			y = self.raw_model(self.X)[..., :-1]
			y_hat = self.model(self.X)[..., :-1]

		for i in np.random.randint(0, self.X.shape[0], 5):
			plt.figure()
			plt.plot(y[i], label="Raw")
			plt.plot(y_hat[i], label="Aggregated")
			plt.legend()

		plt.show()