import os
import typing
import unittest
from abc import abstractmethod, ABC

import numpy as np

from core.utils.research.data.prepare.swg.abstract_swe import SampleWeightExporter
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.standardized_swe import StandardizedSampleWeightExporter
from lib.utils.logger import Logger


class AbstractSampleWeightGeneratorTest(unittest.TestCase, ABC):

	@abstractmethod
	def _init_generator(self) -> AbstractSampleWeightGenerator:
		pass

	@abstractmethod
	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		pass

	def _init_exporter(self, data_path: str, export_path: str, generator: AbstractSampleWeightGenerator) -> SampleWeightExporter:
		return StandardizedSampleWeightExporter(
			input_paths=self._get_input_paths(data_path),
			export_path=export_path,
			generator=generator,
			target_std=1.0,
			target_mean=2
		)

	def _init_datapath(self) -> str:
		return "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/7/train"

	def __print_sample(self):
		for filename in os.listdir(self.export_path):
			np_weights = np.load(os.path.join(self.export_path, filename))
			Logger.info("Weights Sample:")
			Logger.info(np_weights)
			break

	def setUp(self):
		self.generator = self._init_generator()
		self.data_path = self._init_datapath()
		self.export_path = os.path.join(self.data_path, "w")
		self.exporter = self._init_exporter(self.data_path, self.export_path, self.generator)

	def test_functionality(self):
		# self.__print_sample()
		self.exporter.start()
		self.__print_sample()

