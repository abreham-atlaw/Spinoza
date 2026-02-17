


import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from core import Config
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.losses import ProximalMaskedLoss3, ProximalMaskedLoss, ProximalMaskedLoss4
from lib.utils.fileio import load_json


class ProximalMaskedLoss3Test(unittest.TestCase):

	def setUp(self):
		self.loss = ProximalMaskedLoss4.load(
			bounds=DataPrepUtils.apply_bound_epsilon(load_json(os.path.join(Config.BASE_DIR, "res/bounds/05.json"))),
			softmax=True,
			s=0.15,
			path=os.path.join(Config.BASE_DIR, "res/losses/pml3_2.json"),
		)
		self.old = ProximalMaskedLoss3.load(
			bounds=DataPrepUtils.apply_bound_epsilon(load_json(os.path.join(Config.BASE_DIR, "res/bounds/05.json"))),
			softmax=True,
			path=os.path.join(Config.BASE_DIR, "res/losses/pml3_2.json"),
		)
		self.y = torch.from_numpy(np.load(os.path.join(Config.BASE_DIR, "temp/Data/prepared/7/train/y/1751195327.143124.npy")).astype(np.float32))[:, :-1]

	def test_plot_mask(self):
		SAMPLES = 5
		idxs = torch.randint(self.loss.mask.shape[0], (SAMPLES,))
		idxs = np.array([31, 34, 45, 21, 8, 54])

		for i in idxs:
			plt.figure()
			plt.grid(True)
			plt.plot(np.arange(self.old.mask.shape[1]), self.old.mask[i].numpy(), label="OLD")
			plt.plot(np.arange(self.loss.mask.shape[1]), self.loss.mask[i].numpy(), label="LOSS")
			plt.legend()
			plt.title(f"i={i}")
		plt.show()
