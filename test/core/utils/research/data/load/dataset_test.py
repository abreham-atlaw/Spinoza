import os
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from core import Config
from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.swg import \
	CompletenessLassSampleWeightGenerator
from core.utils.research.data.prepare.swg.manipulate import StandardizeSampleWeightManipulator
from core.utils.research.data.prepare.swg.swg_pipeline import SampleWeightGeneratorPipeline


class BaseDatasetTest(unittest.TestCase):

	def setUp(self):
		self.dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/train"
			],
			num_files=3,
			load_weights=True,
			return_weights=False
		)

	def test_functionality(self):

		X, y, w = self.dataset[15]

		print(X, y, w)
		self.assertEqual(X.shape[0], 1148)
		self.assertEqual(y.shape[0], 432)
		self.assertIsNotNone(w)

	def test_shuffle(self):
		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
			],
			return_weights=False
		)

		X_0, y_0 = dataset[54]

		dataset.shuffle()
		X_1, y_1 = dataset[54]

		dataset.shuffle()
		X_2, y_2 = dataset[54]

		for i, (X, y) in enumerate(zip([X_0, X_1, X_2], [y_0, y_1, y_2])):
			for j, (X_o, y_o) in enumerate(zip([X_0, X_1, X_2], [y_0, y_1, y_2])):
				if i == j:
					continue
				self.assertFalse(torch.all(X == X_o))
				self.assertFalse(torch.all(y == y_o))

	def test_dataloader(self):
		dataloader = DataLoader(
			dataset=self.dataset,
			batch_size=6
		)

		self.dataset.shuffle()

		for X, y, w in dataloader:
			self.assertIsNotNone(X)

	def test_lass_load(self):
		dataset = BaseDataset(
			root_dirs=[
				os.path.join(Config.BASE_DIR, "temp/Data/lass/6/train")
			],
			load_weights=False,
			swg=SampleWeightGeneratorPipeline(
				generators=[
					CompletenessLassSampleWeightGenerator(),
					StandardizeSampleWeightManipulator(
						current_mean=0.0427,
						current_std=0.10,
						target_mean=1.0,
						target_std=0.3
					)
				]
			)
		)

		X, y, w = dataset[0]

		self.assertEqual(X.shape[0], 32)
		self.assertEqual(y.shape[0], 1)
		self.assertEqual(w.shape[0], 1)

