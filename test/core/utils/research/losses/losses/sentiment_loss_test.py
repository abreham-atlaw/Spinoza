import unittest

import torch

from core.utils.research.losses import SentimentLoss


class SentimentLossTest(unittest.TestCase):

	def test_mock_data(self):
		bounds = [-2, -1, 0, 1, 2]
		loss = SentimentLoss(
			bounds=torch.Tensor(bounds),
			softmax=False,
			multi_channel=True,
			bound_neutral=0
		)
		loss1 = SentimentLoss(
			bounds=torch.Tensor(bounds),
			softmax=False,
			multi_channel=True,
			bound_neutral=0,
			d=10
		)
		loss2 = SentimentLoss(
			bounds=torch.Tensor(bounds),
			softmax=False,
			multi_channel=True,
			bound_neutral=0,
			d=10,
			channels_weight=[0, 1, 0]
		)

		y = torch.Tensor([
			[
				[1, 0, 0, 0, 0],
				[0, 1, 0, 0, 0],
				[0, 0, 0, 1, 0],
			]
		])
		y_hat = torch.Tensor([
			[
				[0, 1, 0, 0, 0],
				[0, 0, 0, 1, 0],
				[0, 1, 0, 0, 0]
			]
		])

		l = loss(y_hat, y)
		l1 = loss1(y_hat, y)
		l2 = loss2(y_hat, y)
		self.assertEqual(l, 5/3)
		self.assertEqual(l1, 45/3)
		self.assertEqual(l2, 22)

	def test_batch(self):
		bounds = [-2, -1, 0, 1, 2]

		loss = SentimentLoss(
			bounds=torch.Tensor(bounds),
			softmax=False,
			multi_channel=True,
			bound_neutral=0,
			d=10,
			channels_weight=[0, 1, 0]
		)

		y = torch.Tensor([
			[
				[1, 0, 0, 0, 0],
				[0, 1, 0, 0, 0],
				[0, 0, 0, 1, 0],
			]
		])
		y_hat = torch.Tensor([
			[
				[0, 1, 0, 0, 0],
				[0, 0, 0, 1, 0],
				[0, 1, 0, 0, 0]
			]
		])

		l = loss(torch.concatenate([y_hat for _ in range(10)]), torch.concatenate([y for _ in range(10)]))
		self.assertEqual(l, 5 / 3)
