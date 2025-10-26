from core.utils.research.data.prepare.augmentation import Transformation, VerticalShiftTransformation, \
	GaussianNoiseTransformation
from .transformation_abstract_test import TransformationAbstractTest


class VerticalShiftTransformationTest(TransformationAbstractTest):

	def _init_transformation(self) -> Transformation:
		return GaussianNoiseTransformation(r_std=0.05)

