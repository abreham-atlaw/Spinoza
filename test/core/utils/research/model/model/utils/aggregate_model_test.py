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
				temperature=1.0,
				model=ModelHandler.load(os.path.join(Config.BASE_DIR, "/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-training-cnn-0-it-89-tot_1.zip")).eval()
			),
			seq_len=128
		)
		self.model = AggregateModel(
			model=self.raw_model,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			a=0.95/3
		).eval()

		self.X = torch.from_numpy(np.load("/home/abrehamatlaw/Downloads/1766196639.72487.npy").astype(np.float32))

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

		if len(y.shape) == 2:
			y, y_hat = [torch.unsqueeze(arr, dim=1) for arr in [y, y_hat]]


		for i in np.random.randint(0, self.X.shape[0], 5):
			plt.figure()

			for j in range(y.shape[1]):
				plt.subplot(2, 2, j+1)
				plt.plot(y[i, j], label="Raw")
				plt.plot(y_hat[i, j], label="Aggregated")
				plt.ylim([0, 1])
				plt.legend()

		plt.show()