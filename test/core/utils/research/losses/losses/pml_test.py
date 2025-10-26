import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt

from core import Config
from core.utils.research.eval.mlpl_evaluator.losses.unbatched_pml_loss import UnbatchedProximalMaskedLoss
from core.utils.research.losses import ProximalMaskedLoss


class ProximalMaskedLossTest(unittest.TestCase):

	def setUp(self):
		self.loss = ProximalMaskedLoss(n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1, softmax=True)

	def test_functionality(self):
		classes = (np.arange(5) + 10).astype(np.float32)

		loss_fn = ProximalMaskedLoss(
			n=len(classes),
			softmax=False
		)

		y = torch.from_numpy(np.array([[0, 1, 0, 0, 0]]).astype(np.float32))
		predictions = torch.from_numpy(np.array([
			[[1 if i == j else 0 for j in range(len(classes))]]
			for i in range(len(classes))
		]).astype(np.float32))

		losses = [loss_fn(y, predictions[i]) for i in range(len(classes))]
		self.assertIsNotNone(losses)
		self.assertEqual(losses[1], 0)
		self.assertEqual(losses[0], losses[2])
		self.assertLess(losses[2], losses[3])

		loss = loss_fn(torch.from_numpy(np.repeat(y.numpy(), axis=0, repeats=len(classes))), torch.squeeze(predictions))
		self.assertEqual(loss, torch.mean(torch.Tensor(losses)))

	def test_weights(self):
		classes = (np.arange(5) + 10).astype(np.float32)
		weights = np.array([0.3, 0.6, 0.9, 0.6, 0.3]).astype(np.float32)

		loss_fn = ProximalMaskedLoss(
			n=len(classes),
			softmax=False,
		)
		w_loss_fn = ProximalMaskedLoss(
			n=len(classes),
			softmax=False,
			weights=weights
		)

		y = torch.from_numpy(np.array([
			[0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1]
		]).astype(np.float32))

		predictions = torch.from_numpy(np.array([
			[1, 0, 0, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 0, 0, 1, 0]
		]).astype(np.float32))

		losses = [loss_fn(predictions[i:i+1], y[i:i+1]) for i in range(y.shape[0])]
		w_losses = [w_loss_fn(predictions[i:i+1], y[i:i+1]) for i in range(y.shape[0])]

		print(f"Losses: {losses}")
		print(f"Weighted Losses: {w_losses}")

		self.assertEqual(losses[0], losses[1])
		self.assertEqual(losses[0], losses[2])

		self.assertGreater(w_losses[1], w_losses[0])
		self.assertGreater(w_losses[1], w_losses[2])
		self.assertGreater(w_losses[0], w_losses[2])

	def test_actual(self):

		classes = Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		weights = Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_WEIGHTS

		loss_fn = ProximalMaskedLoss(
			n=len(classes)+1,
			softmax=False,
			weights=weights
		)

		y = torch.from_numpy(np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/y/1743180011.758194.npy").astype(np.float32))[:, :-1]

		predictions = torch.from_numpy(np.random.random((y.shape[0], y.shape[1])).astype(np.float32))

		loss = loss_fn(predictions, y)

		print(loss)

	def test_unbatched(self):
		classes = Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		weights = Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_WEIGHTS

		loss_fn = ProximalMaskedLoss(
			n=len(classes) + 1,
			softmax=True,
			weights=weights,
			collapsed=False
		)

		y = torch.from_numpy(np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/y/1743180011.758194.npy").astype(
			np.float32))[:, :-1]

		predictions = torch.from_numpy(np.random.random((y.shape[0], y.shape[1])).astype(np.float32))

		loss = loss_fn(predictions, y)

		print(loss)
		self.assertEqual(loss.shape[0], y.shape[0])
		self.assertEqual(len(loss.shape), 1)

	def test_sample_weighted(self):
		classes = Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND

		loss_fn = ProximalMaskedLoss(
			n=len(classes) + 1,
			softmax=True,
			collapsed=True,
			weighted_sample=True
		)

		y = torch.from_numpy(np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/y/1743180011.758194.npy").astype(
			np.float32))[:, :-1]
		w = torch.from_numpy(np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/dp_weights/4/test/1743180011.758194.npy").astype(
			np.float32))

		predictions = torch.from_numpy(np.random.random((y.shape[0], y.shape[1])).astype(np.float32))

		loss = loss_fn(predictions, y, w)

		print(loss)

	def test_plot_mask(self):

		SAMPLES = 5
		idxs = torch.randint(self.loss.mask.shape[0], (SAMPLES,))
		idxs = np.array([34, 45, 21, 8, 54])
		for i in idxs:
			plt.figure()
			plt.grid(True)
			plt.plot(self.loss.mask[i].numpy())
			plt.title(f"i={i}")
		plt.show()
