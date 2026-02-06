import typing
from abc import ABC, abstractmethod

import numpy as np

from core.environment.trade_state import TradeState


class StateTransitionSampler(ABC):

	@abstractmethod
	def sample_next_values(self, state: TradeState, instrument: typing.Tuple[str, str]) -> np.ndarray:
		pass
