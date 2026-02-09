import os.path
import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt

from core import Config
from core.utils.research.model.model.ensemble.boost.confidence_boost_model import ConfidenceBoostModel
from lib.utils.torch_utils.model_handler import ModelHandler


class ConfidenceBoostModelTest(unittest.TestCase):

	def setUp(self):
		self.pre_model = ModelHandler.load(os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-training-cnn-1-it-89-tot.0.zip"))
		self.post_model = ModelHandler.load(os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-training-cnn-8-it-89-tot.zip"))
		self.confidence_model = ModelHandler.load(os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-confidence-training-model-1-it-0.zip"))
		self.boost_model = ConfidenceBoostModel(
			pre_model=self.pre_model,
			post_model=self.post_model,
			confidence_model=self.confidence_model,
			confidence_range=(0.15, 0.20),
			softmax=True
		)

	def test_call(self):
		X = torch.from_numpy(np.load(os.path.join(Config.BASE_DIR, "temp/Data/simulation_simulator_data/08/test/X/1765787442.2392.npy")).astype(np.float32))
		with torch.no_grad():
			y_hat = self.boost_model(X)

		for i in np.random.randint(0, X.shape[0], 5):
			plt.figure()

			plt.subplot(1, 2, 1)
			plt.plot(X[i].transpose(1, 0))

			plt.subplot(1, 2, 2)
			plt.plot(y_hat[i].transpose(1, 0))
			plt.ylim([0, 1])

		plt.show()
