from typing import Optional

import numpy as np

from ..stm import ShortTermMemory
from .evaluator import StochasticMemoryEvaluator
from lib.utils.logger import Logger


class StochasticShortTermMemory(ShortTermMemory):

	def __init__(
			self,
			evaluator: StochasticMemoryEvaluator,
			*args,
			match_threshold: float = None,
			**kwargs,
	):
		super().__init__(
			*args,
			**kwargs,
			matcher=None
		)
		self._evaluator = evaluator
		self._match_threshold = match_threshold

	def __evaluate(self, cue, memory) ->float:
		if isinstance(memory, ShortTermMemory.EmptyMemory):
			return np.inf
		return self._evaluator(cue, memory)

	def recall(self, cue) -> Optional[object]:

		values = [
			self.__evaluate(cue, memory)
			for memory in self._memories
		]

		Logger.info(f"Values: {values}")

		min_idx = np.argmin(values)

		if (self._match_threshold is not None and values[min_idx] > self._match_threshold) or isinstance(self._memories[min_idx], ShortTermMemory.EmptyMemory):
			return None

		return self._export_memory(self._memories[min_idx])
