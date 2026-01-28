import os
import unittest

import numpy as np
from matplotlib import pyplot as plt

from core import Config
from core.utils.research.data.prepare.confidence_preparer import ConfidencePreparer
from core.utils.research.losses import ProximalMaskedLoss
from lib.utils.logger import Logger
from lib.utils.torch_utils.model_handler import ModelHandler


class ConfidencePreparerTest(unittest.TestCase):

	def setUp(self):
		self.data_path = os.path.join(Config.BASE_DIR, "temp/Data/simulation_simulator_data/08/test")
		self.export_path = os.path.join(Config.BASE_DIR, "temp/Data/confidence_data/00")

		Logger.info(f"Cleaning export path: {self.export_path}")
		os.system("rm -fr '{}'".format(self.export_path))

		self.model = ModelHandler.load(os.path.join(Config.BASE_DIR, "abrehamalemu-spinoza-training-cnn-1-it-89-tot.0.zip"))
		self.loss = ProximalMaskedLoss(
			n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
			multi_channel=True,
			collapsed=False
		)
		self.preparer = ConfidencePreparer(
			data_path=self.data_path,
			export_path=self.export_path,
			model=self.model,
			loss=self.loss
		)

	def test_preparation(self):
		self.preparer.start()

		filenames = os.listdir(os.path.join(self.export_path, "X_Encoder"))

		for i, idx in enumerate(np.random.randint(0, len(filenames), 10)):

			x_encoder, x_decoder, y = [
				np.load(os.path.join(self.export_path, dir_name, filenames[idx]))
				for dir_name in os.listdir(self.export_path)
			]

			plt.figure()
			plt.subplot(1, 2, 1)
			plt.plot(x_encoder[0].transpose(1, 0))
			plt.subplot(1, 2, 2)
			plt.plot(x_decoder[0, :-1].transpose(1, 0))
			plt.title(f"Confidence: {y[0]}")

		plt.show()
