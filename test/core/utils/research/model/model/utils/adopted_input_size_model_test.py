import unittest

import torch

from core.utils.research.model.model.utils import AdoptedInputSizeModel
from lib.utils.logger import Logger
from lib.utils.torch_utils.model_handler import ModelHandler


class AdoptedInputSizeModelTest(unittest.TestCase):

	def setUp(self):
		self.model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-training-cnn-0-it-53-tot.zip").eval()
		self.adopted_model = AdoptedInputSizeModel(
			input_size=1024+124,
			model=self.model
		)

	def test_call(self):
		x = torch.rand((5, self.model.input_size[1]))
		x_adopted = torch.concatenate(
			(
				torch.zeros(x.shape[0], self.adopted_model.input_size[1] - self.model.input_size[1]),
				x
			),
			dim=1
		)

		Logger.info(f"x shape: {x.shape}")
		Logger.info(f"x_adopted shape: {x_adopted.shape}")

		y = self.model(x)
		y_adopted = self.adopted_model(x_adopted)

		self.assertTrue(torch.allclose(y, y_adopted))
