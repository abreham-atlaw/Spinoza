from abc import ABC
from copy import deepcopy
from typing import List, Any

import numpy as np

from core import Config
from core.agent.action import Action
from core.di import AgentUtilsProvider
from core.environment.trade_state import TradeState
from lib.rl.agent.mca import ReflexMonteCarloAgent
from lib.rl.environment import ModelBasedState
from lib.utils.stm import StochasticShortTermMemory
from .stm.reflex import TraderReflexMemoryEvaluator
from ..drmca import TraderDeepReinforcementMonteCarloAgent


class ReflexAgent(
	ReflexMonteCarloAgent,
	TraderDeepReinforcementMonteCarloAgent,
	ABC
):

	def __init__(
			self,
			*args,
			**kwargs
	):
		super().__init__(
			*args,
			reflex_stm=AgentUtilsProvider.provide_reflex_stm(),
			**kwargs
		)

	def _generate_actions(self, state: TradeState) -> List[object]:
		if state.pre_computation:
			return [None]
		return super()._generate_actions(state)

	def _get_possible_states(self, state: TradeState, action: Action) -> List[TradeState]:

		if state.pre_computation:
			assert action is None
			states = super()._get_possible_states(state, None)
			for state in states:
				state.pre_computation = False
			return states

		new_state = deepcopy(state)
		self._simulate_action(new_state, action)
		new_state.pre_computation = True
		return [new_state]

	def _get_expected_transition_probability_distribution(
			self, initial_states: List[TradeState], action: List[Any], final_states: List[TradeState]
	) -> List[float]:
		probabilities = np.ones(len(initial_states))

		pre_computation_mask = np.array([state.pre_computation for state in initial_states])
		if np.any(pre_computation_mask):
			probabilities[pre_computation_mask] = super()._get_expected_transition_probability_distribution(
				list(np.array(initial_states)[pre_computation_mask]),
				list(np.array(action)[pre_computation_mask]),
				list(np.array(final_states)[pre_computation_mask])
			)

		return probabilities

	def _monte_carlo_tree_search(self, state: TradeState) -> None:
		super()._monte_carlo_tree_search(state)
		state.pre_computation = False
