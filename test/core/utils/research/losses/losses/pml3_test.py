import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from core import Config
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.losses import ProximalMaskedLoss3, ProximalMaskedLoss
from lib.utils.fileio import load_json


class ProximalMaskedLoss3Test(unittest.TestCase):

	def setUp(self):
		self.loss = ProximalMaskedLoss3(
			bounds=DataPrepUtils.apply_bound_epsilon(load_json(os.path.join(Config.BASE_DIR, "res/bounds/05.json"))),
			softmax=True,
		)
		self.pml = ProximalMaskedLoss(
			n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
			softmax=True
		)
		self.y = torch.from_numpy(np.load(os.path.join(Config.BASE_DIR, "temp/Data/prepared/7/train/y/1751195327.143124.npy")).astype(np.float32))[:, :-1]

	def test_mock(self):
		classes = (1 + 0.05*(np.arange(5)-2)).astype(np.float32)

		loss_fn = ProximalMaskedLoss3(
			bounds=classes,
			softmax=False
		)

		y = torch.from_numpy(np.array([[0, 1, 0, 0, 0]]).astype(np.float32))
		predictions = torch.from_numpy(np.array([
			[[1 if i == j else 0 for j in range(len(classes))]]
			for i in range(len(classes))
		]).astype(np.float32))

		losses = torch.Tensor([loss_fn(y, predictions[i]) for i in range(len(classes))])
		self.assertIsNotNone(losses)
		self.assertEqual(losses[1], torch.min(losses))
		self.assertEqual(losses[3], torch.max(losses))
		self.assertLess(losses[0], losses[2])

		loss = loss_fn(torch.from_numpy(np.repeat(y.numpy(), axis=0, repeats=len(classes))), torch.squeeze(predictions))
		self.assertEqual(loss, torch.mean(torch.Tensor(losses)))

	def test_actual(self):

		y_hat = torch.from_numpy(np.random.random((self.y.shape[0], self.y.shape[1])).astype(np.float32))
		loss = self.loss(y_hat, self.y)
		print(loss)

	def test_plot_mask(self):
		SAMPLES = 5
		idxs = torch.randint(self.loss.mask.shape[0], (SAMPLES,))
		idxs = np.array([31, 34, 45, 21, 8, 54])

		for i in idxs:
			plt.figure()
			plt.grid(True)
			plt.plot(np.arange(self.loss.mask.shape[1]), self.loss.mask[i].numpy(), label="PML3")
			plt.plot(np.arange(self.pml.mask.shape[1]), self.pml.mask[i].numpy(), label="PML")
			plt.legend()
			plt.title(f"i={i}")
		plt.show()
