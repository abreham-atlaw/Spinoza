import os.path
import unittest

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from core import Config
from core.utils.research.model.model.utils import AggregateModel, WrappedModel, TemperatureScalingModel
from lib.utils.torch_utils.model_handler import ModelHandler


class AggregateModelTest(unittest.TestCase):


	def setUp(self):
		self.raw_model =ModelHandler.load(os.path.join(Config.BASE_DIR, "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra.zip")).eval()
		self.wrapped_model = WrappedModel(
			TemperatureScalingModel(
				temperature=1.0,
				model=self.raw_model
			),
			seq_len=128
		)
		self.model = AggregateModel(
			model=self.wrapped_model,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			a=0.98/3,
			temperature=1e-5
		).eval()
		self.softmax_model = AggregateModel(
			model=self.raw_model,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			a=0.98/3,
			temperature=1e-5,
			softmax=True
		)


		self.X = torch.from_numpy(np.load("/home/abrehamatlaw/Downloads/1768046021.603058.npy").astype(np.float32))

	def test_aggregate(self):
		with torch.no_grad():
			y = self.wrapped_model(self.X)[...,0 , :-1]
			y_hat = self.model.aggregate(y)

		for i in np.random.randint(0, self.X.shape[0], 5):
			plt.figure()
			plt.plot(y[i], label="Raw")
			plt.plot(y_hat[i], label="Aggregated")
			plt.legend()

		plt.show()

	def test_forward(self):

		with torch.no_grad():
			y = self.wrapped_model(self.X)[..., :-1]
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

	def test_aggregate_consistency(self):
		with torch.no_grad():
			y_hat = torch.stack([
				self.model(self.X)[..., :-1]
				for _ in range(10)
			], dim=0)

		for j in np.random.randint(0, y_hat.shape[1], 3):
			plt.figure()

			for i in range(y_hat.shape[0]):
				plt.plot(y_hat[i, j, 0], label=f"Call: {i}")
			plt.legend()
		plt.show()

	def test_softmax(self):

		softmax = nn.Softmax(dim=-1)
		with torch.no_grad():
			y = self.model(self.X)[..., :-1]
			y_hat = softmax(self.softmax_model(self.X)[..., :-1])

		if len(y.shape) == 2:
			y, y_hat = [torch.unsqueeze(arr, dim=1) for arr in [y, y_hat]]


		for i in np.random.randint(0, self.X.shape[0], 5):
			plt.figure()

			for j in range(y.shape[1]):
				plt.subplot(2, 2, j+1)
				plt.plot(y[i, j], label="Non-Softmax")
				plt.plot(y_hat[i, j], label="Softmax")
				plt.ylim([0, 1])
				plt.legend()

		plt.show()
