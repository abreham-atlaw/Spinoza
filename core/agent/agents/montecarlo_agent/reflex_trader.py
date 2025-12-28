from abc import ABC
from copy import deepcopy
from typing import List

from core.agent.action import Computation, Action
from core.environment.trade_state import TradeState
from ..drmca import TraderDeepReinforcementMonteCarloAgent


class ReflexAgent(TraderDeepReinforcementMonteCarloAgent, ABC):

	def _generate_actions(self, state: TradeState) -> List[object]:
		if state.pre_computation:
			return [Computation()]
		return super()._generate_actions(state)

	def _get_possible_states(self, state: TradeState, action: Action) -> List[TradeState]:

		if isinstance(action, Computation):
			states = super()._get_possible_states(state, None)
			for state in states:
				state.pre_computation = False
			return states

		new_state = deepcopy(state)
		self._simulate_action(new_state, action)
		new_state.pre_computation = True
		return [new_state]
