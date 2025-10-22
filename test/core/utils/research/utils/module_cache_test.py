import unittest

import torch

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model.lass3 import Lass3HorizonModel
from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model.lass3.transformer import Lass3Transformer
from core.utils.research.utils.module_cache import ModuleCache
from lib.utils.devtools import performance
from lib.utils.logger import Logger
from lib.utils.torch_utils.model_handler import ModelHandler


class ModuleCacheTest(unittest.TestCase):

	def setUp(self):
		self.cache = ModuleCache()
		self.model: Lass3HorizonModel = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-lass-training-cnn-10-it-5-tot_2.zip")
		self.model.eval()

	def test_cache(self):

		batches = [
			((torch.rand((128, 6)),), torch.rand((1, 1)))
			for _ in range(10)
		]

		for x, y in batches:
			self.cache.store(x, y)

		for x, y in batches:
			cached = self.cache.retrieve(x)
			self.assertTrue(torch.all(torch.eq(cached, y)))

	def test_model_caching(self):

		def generate_xs():
			x = torch.zeros((10_000, 2, 128))
			x[:, 0, :] = torch.rand((x.shape[0], x.shape[2]))

			xs = [x.clone() for _ in range(64)]
			for i in range(1, len(xs)):
				xs[i][:, 1, -i:] = torch.rand((x.shape[0], i))
			return xs

		def call_model():
			with torch.no_grad():
				y = [self.model(x) for x in Xs]
			return y

		Xs = generate_xs()

		Logger.info(f"Running non-cached inference")
		ys = performance.track_performance("model", call_model)

		self.model.model.encoder_block.set_caching(True)
		Logger.info(f"Running cached inference")
		cached = performance.track_performance("cached", call_model)

		self.assertTrue(all(torch.all(torch.eq(y, cached_y)) for y, cached_y in zip(ys, cached)))

		print(performance.durations)
