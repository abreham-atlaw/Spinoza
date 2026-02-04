import typing
from abc import ABC, abstractmethod


class StochasticMemoryEvaluator(ABC):

	@abstractmethod
	def evaluate(self, cue: typing.Any, memory: typing.Any) -> float:
		pass

	def __call__(self, cue, memory):
		return self.evaluate(cue, memory)
