import typing

import matplotlib.pyplot as plt
import numpy as np
from lesscpy.lessc.utility import blocksearch

from lib.utils.logger import Logger


class SinusoidalDecomposer:

	def __init__(
			self,
			fqs: typing.Tuple[int, int, int] = (0.1, 20, 50),
			shifts: typing.Tuple[int, int, int] = (0, 6, 50),
			layer_indifference_threshold: float = 0.05,
			min_block_size: int = 512,
			block_layers: int = 10,
			plot_progress: bool = True,
			collapse_output: bool = True
	):
		self.__layer_indifference_threshold = layer_indifference_threshold
		self.__initial_fqs = np.linspace(*fqs)
		self.__initial_shifts = np.linspace(*shifts)
		self.__min_block_size = min_block_size
		self.__block_layers = block_layers
		self.__plot_progress = plot_progress
		self.__collapse_output = collapse_output

	@staticmethod
	def __wave(f: float, s: float, l: float) -> np.ndarray:
		return np.sin(np.linspace(0, 2*np.pi, l)*f + s)

	@staticmethod
	def __v_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
		return (((source - np.mean(source)) * np.std(target))/np.std(source)) + np.mean(target)

	@staticmethod
	def __loss(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
		return np.sum((y_hat - y)**2, axis=-1)

	def __generate_waves_grid(self, fqs: typing.List[float], shifts: typing.List[float], x: np.ndarray) -> np.ndarray:

		waves = np.array([
			[
				self.__v_align(self.__wave(f, s, x.shape[0]), x)
				for s in shifts
			]
			for f in fqs
		])

		return waves

	def __evaluate_configs(self, fqs: typing.List[float], shifts: typing.List[float], x: np.ndarray) -> np.ndarray:
		y = self.__generate_waves_grid(fqs, shifts, x)
		loss = self.__loss(y, x)
		return loss

	def __optimize_layer(
			self,
			fqs: typing.List[float],
			shifts: typing.List[float],
			x: np.ndarray,
			previous_loss: float = None
	) -> typing.Tuple[float, float]:
		loss = self.__evaluate_configs(fqs, shifts, x)

		min_loss_idx = np.argmin(loss)
		min_f_idx, min_s_idx = min_loss_idx // loss.shape[1], min_loss_idx % loss.shape[1]

		min_loss = loss[min_f_idx, min_s_idx]
		f, s = fqs[min_f_idx], shifts[min_s_idx]

		if previous_loss is not None and min_loss > previous_loss*(1-self.__layer_indifference_threshold):
			return None

		fqs, shifts = [
			np.linspace(config[max(idx - 1, 0)], config[min(idx + 1, len(config) - 1)], len(config))
			for config, idx in [(fqs, min_f_idx), (shifts, min_s_idx)]
		]

		new_configs = self.__optimize_layer(fqs, shifts, x, min_loss)
		if new_configs is None:
			return f, s
		return new_configs

	def __optimize_block(self, x: np.ndarray, layers: int = None) -> np.ndarray:
		if layers is None:
			layers = self.__block_layers
		Logger.info(f"Optimizing {layers} layers...", end="\r")

		f, s = self.__optimize_layer(self.__initial_fqs, self.__initial_shifts, x)

		y = np.expand_dims(self.__v_align(self.__wave(f, s, x.shape[0]), x), axis=0)

		if layers == 1:
			return y

		new_layers = self.__optimize_block(x-y[0], layers-1)
		y = np.concatenate((
			y, new_layers
		), axis=0)

		return y

	def __split_and_optimize(self, x: np.ndarray, block_size: int) -> np.ndarray:
		Logger.info(f"Splitting and Optimizing with block_size={block_size}")

		y = None

		for i in range(0, int(np.ceil(x.shape[0] / block_size))):
			block = np.sum(self.__optimize_block(
				x[i*block_size: (i+1)*block_size]
			), axis=0)
			if y is None:
				y = block
				continue
			block += y[-1] - block[0]
			y = np.concatenate((y, block), axis=-1)

		y = self.__v_align(y, x)

		return y

	def __optimize(self, x: np.ndarray) -> np.ndarray:

		diff = x
		y = np.zeros((0, x.shape[0]))

		blocks = 1
		block_size = x.shape[0]

		while block_size > self.__min_block_size:

			step_y = np.expand_dims(self.__split_and_optimize(diff, block_size), axis=0)
			y = np.concatenate((y, step_y), axis=0)

			if self.__plot_progress:
				plt.figure(figsize=(15, 3.75))

				plt.subplot(1, 2, 1)
				plt.plot(diff)
				plt.plot(np.sum(step_y, axis=0))

				plt.subplot(1, 2, 2)
				plt.plot(x)
				plt.plot(np.sum(y, axis=0))

				plt.show()

			diff = diff - np.sum(step_y, axis=0)

			blocks += 1
			block_size = int(np.ceil(x.shape[0] / blocks))

		return y

	def decompose(self, x: np.ndarray) -> np.ndarray:
		Logger.info(f"Starting Optimization...")
		y = self.__optimize(x)
		if self.__collapse_output:
			y = np.sum(y, axis=0)
		return y
