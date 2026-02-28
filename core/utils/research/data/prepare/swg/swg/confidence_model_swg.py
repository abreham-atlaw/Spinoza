import numpy as np
import torch

from core.utils.research.model.model.savable import SpinozaModule
from .basic_swg import BasicSampleWeightGenerator


class ConfidenceModelSampleWeightGenerator(BasicSampleWeightGenerator):


	def __init__(
			self,
			model: SpinozaModule,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.model = model.eval()

	def _generate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		with torch.no_grad():
			confidence = self.model(torch.from_numpy(X.astype(np.float32))).numpy().reshape((X.shape[0],))
		return 1 / confidence
