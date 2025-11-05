import numpy as np

from core.utils.research.data.prepare.swg.swg import BasicSampleWeightGenerator


class CompletenessLassSampleWeightGenerator(BasicSampleWeightGenerator):

	def _generate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		return 1 / np.sum(X[:, 1] == 0, axis=1)
