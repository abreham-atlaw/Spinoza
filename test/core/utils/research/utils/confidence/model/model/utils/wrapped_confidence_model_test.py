import os.path
import unittest

import numpy as np
import torch

from core import Config
from core.utils.research.utils.confidence.model.model.utils import WrappedConfidenceModel
from lib.utils.torch_utils.model_handler import ModelHandler


class WrappedConfidenceModelTest(unittest.TestCase):

	def setUp(self):
		self.model = WrappedConfidenceModel(
			core_model=ModelHandler.load(os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-training-cnn-1-it-89-tot.0.zip")),
			confidence_model=ModelHandler.load(os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-confidence-training-model-1-it-0.zip"))
		).eval()
		self.X = torch.from_numpy(np.load(os.path.join(Config.BASE_DIR, "temp/Data/simulation_simulator_data/08/train/X/1765787442.18596.npy")).astype(np.float32))

	def test_functionality(self):
		with torch.no_grad():
			y_hat = self.model(self.X)

		self.assertEqual(y_hat.shape, (self.X.shape[0], 1))



