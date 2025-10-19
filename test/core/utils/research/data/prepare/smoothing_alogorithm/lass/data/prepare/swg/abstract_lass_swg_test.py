from abc import ABC

import os

from core import Config
from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest


class AbstractLassSampleWeightGeneratorTest(AbstractSampleWeightGeneratorTest, ABC):

	def _init_datapath(self) -> str:
		return os.path.join(Config.BASE_DIR, "temp/Data/lass/6/train")
