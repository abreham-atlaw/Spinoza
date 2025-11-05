import unittest

import torch

from core.utils.research.model.layers import PadOverlayCombiner, OverlaysCombiner


class PadOverlayCombinerTest(unittest.TestCase):

	def setUp(self):
		self.combiner = PadOverlayCombiner()

	def test_call(self):
		x = [
			torch.rand((10, n))
			for n in [7, 9, 10]
		]

		y = self.combiner(x)
		self.assertEqual(y.shape, (10, 3, 10))
		for i in range(3):
			self.assertEqual(y[0, 0, i], x[0][0, 0])

		print(y)
