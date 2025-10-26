import os.path
import typing
from abc import abstractmethod, ABC

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from core.utils.research.model.model.transformer import DecoderBlock
from lib.utils.logger import Logger
from .model_analyzer import ModelAnalyzer
from .utils.layer_utils import LayerUtils
from .utils.plot_utils import PlotUtils


class AbstractLayerModelAnalyzer(ModelAnalyzer, ABC):

	TRANSPOSED_LAYERS = [
		DecoderBlock,
	]

	def __init__(
			self,
			*args,
			layers: typing.Dict[str, nn.Module] = None,
			skip_identity: bool = True,
			plot: bool = True,
			export_path: str = "./",
			transpose_modules: bool = True,
			transposable_modules: typing.List[typing.Type] = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__plot = plot

		if layers is None:
			layers = LayerUtils.get_layers(self._model, skip_identity=skip_identity)

		self.__layers = layers
		self.__export_path = os.path.abspath(export_path)

		if transposable_modules is None:
			transposable_modules = AbstractLayerModelAnalyzer.TRANSPOSED_LAYERS
		self.__transposable_modules = transposable_modules
		self.__transpose_modules = transpose_modules

	@abstractmethod
	def _get_analysis(self, model: nn.Module, layer: nn.Module, X: torch.Tensor, y:torch.Tensor, w: torch.Tensor) -> torch.Tensor:
		pass

	def _prepare_plot(self, analysis: torch.Tensor) -> torch.Tensor:
		return analysis

	def _get_plot_title(self, name: str, layer: nn.Module) -> str:
		return f"{self._get_export_prefix()}-{name}({layer.__class__.__name__})"

	def _prepare_export(self, analysis: torch.Tensor) -> torch.Tensor:
		return analysis

	def _plot_analysis(self, analysis: torch.Tensor, title: str):
		PlotUtils.plot(y=analysis, title=title)

	def _plot_layer_analysis(self, name: str, layer: nn.Module, analysis: torch.Tensor):
		Logger.info(f"Plotting layer \"{name}\" analysis")
		self._plot_analysis(analysis, title=self._get_plot_title(name, layer))

	def _get_export_prefix(self) -> str:
		return self.__class__.__name__

	def _get_export_path(self, name: str, layer: nn.Module) -> str:
		return os.path.join(self.__export_path, f"{self._get_export_prefix()}-{name}.npy")

	def _export_array(self, arr: torch.Tensor, path: str):
		Logger.info(f"Exporting to {path}...")
		arr = arr.detach().numpy()
		np.save(path, arr)

	def _export_layer_analysis(self, name: str, layer: nn.Module, analysis: torch.Tensor):
		Logger.info(f"Exporting layer \"{name}\" analysis...")
		self._export_array(analysis, self._get_export_path(name, layer))

	def _is_transposable(self, layer: nn.Module, model: nn.Module) -> bool:
		if not self.__transpose_modules:
			return False

		transposable_parents = list(filter(
			lambda layer: True in [isinstance(layer, layer_cls) for layer_cls in self.__transposable_modules],
			model.modules()
		))

		transposable_modules = []
		for parent in transposable_parents:
			transposable_modules.extend(list(parent.modules()))
		transposable_modules = list(set(transposable_modules))
		return layer in transposable_modules

	@staticmethod
	def _transpose_analysis(analysis: torch.Tensor) -> torch.Tensor:
		return torch.transpose(analysis, -1, -2)

	def _analyze_layer(self, model: nn.Module, name: str, layer: nn.Module, X: torch.Tensor, y:torch.Tensor, w: torch.Tensor):
		Logger.info(f"Analyzing layer \"{name}\"...")

		analysis = self._get_analysis(model, layer, X, y, w)

		if self._is_transposable(layer, model):
			Logger.info(f"Transposing \"{name}\" analysis")
			analysis = self._transpose_analysis(analysis)

		self._export_layer_analysis(name, layer, self._prepare_export(analysis))

		if self.__plot:
			self._plot_layer_analysis(name, layer, self._prepare_plot(analysis))

		Logger.success(f"Layer \"{name}\" analysis complete!")

	def _analyze(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor):

		Logger.info(f"Analyzing {len(self.__layers)} layers...")

		for name, layer in self.__layers.items():
			try:
				self._analyze_layer(model, name, layer, X, y, w)
			except (ValueError, RuntimeError, TypeError) as ex:
				Logger.error(f"Failed to analyze \"{name}\": {ex}")

		if self.__plot:
			plt.show()
