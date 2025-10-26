import random

import numpy as np
import torch
from torch import nn

from core.utils.research.utils.model_analysis import f as F
from .abstract_layer_ma import AbstractLayerModelAnalyzer


class LayerOutputModelAnalyzer(AbstractLayerModelAnalyzer):

	def __init__(
			self,
			*args,
			plot_samples: int = 2,
			plot_samples_random_state: int = 42,
			**kwargs
	):
		super(LayerOutputModelAnalyzer, self).__init__(*args, **kwargs)
		self.__plot_samples_size = plot_samples
		self.__plot_samples_random_state = plot_samples_random_state

	def _process_output(self, output: torch.Tensor) -> torch.Tensor:
		return output

	def _prepare_plot(self, analysis: np.ndarray) -> torch.Tensor:
		return self.__get_plot_samples(analysis)

	def _get_analysis(
			self,
			model: nn.Module,
			layer: nn.Module,
			X: torch.Tensor,
			y: torch.Tensor,
			w: torch.Tensor
	) -> torch.Tensor:
		output = F.get_layer_output(model, X, layer)
		output = self._process_output(output)
		return output

	def __get_plot_samples(self, X: torch.Tensor) -> torch.Tensor:
		idxs = torch.arange(X.shape[0]).tolist()
		r = random.Random(self.__plot_samples_random_state)
		r.shuffle(idxs)
		return X[idxs[:self.__plot_samples_size]]
