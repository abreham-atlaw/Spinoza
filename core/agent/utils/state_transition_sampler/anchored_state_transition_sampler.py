import typing

import numpy as np

from core.environment.trade_state import TradeState
from .basic_state_transition_sampler import BasicStateTransitionSampler

class AnchoredStateTransitionSampler(BasicStateTransitionSampler):

	def __init__(
			self,
			*args,
			anchor_channel: str = "c",
			log: bool = True,
			**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.__anchor_channel_idx: int = self._channels.index(anchor_channel)
		self.__log = log

	def _get_possible_values(self, original_values: np.ndarray) -> np.ndarray:
		y = self._bounds
		if self.__log:
			y = np.exp(y)

		possible_values = original_values[[self.__anchor_channel_idx]*original_values.shape[0], -1:] * y
		return possible_values
