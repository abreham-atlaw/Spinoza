from core.utils.research.data.prepare.augmentation import Transformation, VerticalShiftTransformation, \
	VerticalStretchTransformation
from .transformation_abstract_test import TransformationAbstractTest


class VerticalStretchTransformationTest(TransformationAbstractTest):

	def _init_transformation(self) -> Transformation:
		return VerticalStretchTransformation(alpha=1.1)
