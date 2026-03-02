import os
import unittest

import numpy as np
import torch

from core import Config
from core.utils.research.model.model.utils import TPTitanModel
from lib.utils.torch_utils.model_handler import ModelHandler


class TPTitanModelTest(unittest.TestCase):


	def setUp(self):
		self.model = TPTitanModel(
			model=ModelHandler.load(os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-training-cnn-2-it-98-tot.zip")),
			extra_len=12
		)
		self.X = torch.from_numpy(np.load(os.path.join(Config.BASE_DIR, "temp/Data/drmca_export/01/X/1772112570.140877.npy")).astype(np.float32))

	def test_call(self):
		with torch.no_grad():
			y_hat = self.model(self.X)
		print(y_hat.shape)
