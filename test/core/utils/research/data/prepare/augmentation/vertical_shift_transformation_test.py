from core.utils.research.data.prepare.augmentation import Transformation, VerticalShiftTransformation
from .transformation_abstract_test import TransformationAbstractTest


class VerticalShiftTransformationTest(TransformationAbstractTest):

	def _init_transformation(self) -> Transformation:
		return VerticalShiftTransformation(shift=0.1)
