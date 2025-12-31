import os
import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt

from core import Config
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.model.model.manual import VelocityManualModel
from lib.rl.agent.dta import TorchModel
from lib.utils.torch_utils.model_handler import ModelHandler


class ManualVelocityModelTest(unittest.TestCase):

	def setUp(self):
		self.model= VelocityManualModel(
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			extra_len=124,
		)
		self.X = torch.from_numpy( np.load("/home/abrehamatlaw/Downloads/1765387111.711033.npy").astype(np.float32))

	def test_call(self):

		with torch.no_grad():
			y = self.model(self.X)[..., :-1]

		b = DataPrepUtils.apply_bound_epsilon(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND)
		y_v = self.X[..., -125] * torch.sum(y * b, dim=-1)
		for i in np.random.randint(0, y.shape[0], 10):
			plt.figure()
			plt.plot(self.X[i, :-124])
			plt.scatter([128], [y_v[i]], color="red")
			plt.show()

	def test_save_and_load(self):

		SAVE_PATH = os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-training-cnn-1-it-69-tot.zip")
		ModelHandler.save(self.model, SAVE_PATH)
		self.model = ModelHandler.load(SAVE_PATH)
		self.test_call()

	def test_torch_model_wrapper(self):
		model = TorchModel(self.model)
