from core.utils.research.data.prepare.augmentation import Transformation, VerticalShiftTransformation, \
	VerticalStretchTransformation, TimeStretchTransformation
from .transformation_abstract_test import TransformationAbstractTest


class TimeStretchTransformationTest(TransformationAbstractTest):

	def _init_transformation(self) -> Transformation:
		return TimeStretchTransformation(max_n=0.75, min_n=0.5)
