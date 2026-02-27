from abc import ABC, abstractmethod

import numpy as np

from core.agent.action import Action
from core.environment.trade_state import TradeState


class TrainingTargetBuilder(ABC):

	@abstractmethod
	def build(self, state: TradeState, action: Action, final_state: TradeState, value: float) -> np.ndarray:
		pass
