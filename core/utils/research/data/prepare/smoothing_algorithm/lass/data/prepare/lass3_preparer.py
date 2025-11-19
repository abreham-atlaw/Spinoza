import numpy as np

from .lass_preparer_2 import LassPreparer2


class Lass3Preparer(LassPreparer2):

	def __init__(
			self,
			*args,
			left_align: bool = False,
			decoder_samples: int = None,
			**kwargs
	):
		super().__init__(
			*args,
			**kwargs,
			trim_extra_gran=True
		)
		self.__left_align = left_align
		self.__decoder_samples = decoder_samples

	@staticmethod
	def _get_sequence_random(sequences: np.ndarray) -> np.random.Generator:
		seed = hash(sequences.tobytes())
		return np.random.default_rng(abs(seed))

	def __apply_mask(self, x: np.ndarray):
		out = np.repeat(x, x.shape[-1], axis=0)

		encoded_mask = np.arange(
			x.shape[-1]
		).reshape(1, -1) >= (x.shape[-1] - np.tile(np.arange(x.shape[-1]), x.shape[0]).reshape(-1, 1))

		if self.__left_align:
			encoded_mask = np.arange(x.shape[-1]).reshape(1, -1) < (np.tile(np.arange(x.shape[-1]), x.shape[0]).reshape(-1, 1))

		value_mask = np.arange(x.shape[-1]).reshape(1, -1) < np.tile(np.arange(x.shape[-1]), x.shape[0]).reshape(-1, 1)

		out[:, 1][encoded_mask] = out[:, 1][value_mask]
		out[:, 1][~encoded_mask] = 0

		return out

	def __select_samples(self, sequences: np.ndarray, dataset: np.ndarray) -> np.ndarray:
		if self.__decoder_samples is None:
			return dataset
		random = self._get_sequence_random(sequences)

		idxs = np.concatenate([
			random.choice(
				np.arange(i*dataset.shape[0] // sequences.shape[0], (i+1)*dataset.shape[0] // sequences.shape[0]),
				size=self.__decoder_samples,
				replace=False
			)
			for i in range(sequences.shape[0])
		])
		return dataset[idxs]

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		x = self._stack_noisy_and_smoothed(sequences)
		x = self.__apply_mask(x)
		x = self.__select_samples(sequences, x)
		return x

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		data = self._stack_noisy_and_smoothed(sequences)
		smoothed = data[:, 1, :]
		y = smoothed.reshape(-1, 1)
		y = self.__select_samples(sequences, y)
		return y
