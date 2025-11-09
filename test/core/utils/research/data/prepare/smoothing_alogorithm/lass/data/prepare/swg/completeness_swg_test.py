import typing

from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.swg import CompletenessLassSampleWeightGenerator
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.manipulate import StandardizeSampleWeightManipulator
from core.utils.research.data.prepare.swg.swg_pipeline import SampleWeightGeneratorPipeline
from test.core.utils.research.data.prepare.swg.swg.basic_swg_test import BasicSampleWeightGeneratorTest
from .abstract_lass_swg_test import AbstractLassSampleWeightGeneratorTest


class CompletenessSampleWeightGeneratorTest(AbstractLassSampleWeightGeneratorTest, BasicSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return SampleWeightGeneratorPipeline(
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
