import typing
import unittest

import os

from core import Config
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.xswg import ConfidenceModelSampleWeightGenerator
from core.utils.research.utils.confidence.model.model.utils import WrappedConfidenceModel
from lib.utils.torch_utils.model_handler import ModelHandler
from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest


class ConfidenceModelSampleWeightGeneratorTest(AbstractSampleWeightGeneratorTest):

	def _init_datapath(self) -> str:
		return os.path.join(Config.BASE_DIR, "temp/Data/simulation_simulator_data/08/train")

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return ConfidenceModelSampleWeightGenerator(
			model=WrappedConfidenceModel(
				core_model=ModelHandler.load(os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-training-cnn-1-it-89-tot.0.zip")),
				confidence_model=ModelHandler.load(os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-confidence-training-model-1-it-0.zip"))
			)
		)

	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		return [
			os.path.join(data_path, axis)
			for axis in ["X"]
		]
