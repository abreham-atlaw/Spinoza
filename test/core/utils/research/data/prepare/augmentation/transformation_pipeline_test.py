from core.utils.research.data.prepare.augmentation import TransformationPipeline, VerticalShiftTransformation, \
	TimeStretchTransformation, Transformation
from .transformation_abstract_test import TransformationAbstractTest


class TransformationPipelineTest(TransformationAbstractTest):

	def _init_transformation(self) -> Transformation:
		return TransformationPipeline([
			VerticalShiftTransformation(1.0),
			TimeStretchTransformation()
		])
