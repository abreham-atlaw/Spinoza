import unittest

import torch
from torch import nn

from core.utils.research.model.layers import IndicatorsSet, Indicators


class IndicatorsSetTest(unittest.TestCase):

	def setUp(self):
		self.indicators = IndicatorsSet(
			channels=[
				(0, 1, 2),
				(3, 4)
			],
			indicators=[
				Indicators(
					delta=[1, 2, 3],
					input_channels=3
				),
				nn.Identity()

			]
		)


	def test_call(self):

		x = torch.reshape(torch.arange(0, 5*5*128), (5, 5, 128))
		y = self.indicators(x)

		self.assertEqual(y.shape, (5, 14, 125))
		self.assertEqual(self.indicators.indicators_len, 14)

		self.assertTrue(torch.all(x[:, (3, 4), 3:] == y[:, (-2, -1), :]))


