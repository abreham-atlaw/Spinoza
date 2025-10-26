import numpy as np

from .simulation_simulator import SimulationSimulator


class SimulationSimulator2(SimulationSimulator):

	def _prepare_sequence_stack(self, x: np.ndarray) -> np.ndarray:
		return x

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		sequences[:, :-1] = self._smoothing_algorithm.apply_on_batch(sequences[:, :-1])
		return super()._prepare_x(sequences)

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		sequences[:, 1:] = self._smoothing_algorithm.apply_on_batch(sequences[:, 1:])
		return super()._prepare_y(sequences)
