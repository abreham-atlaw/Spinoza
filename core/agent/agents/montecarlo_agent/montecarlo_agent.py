from abc import ABC


from core import Config
from core.di import AgentUtilsProvider
from lib.rl.agent import MonteCarloAgent


class TraderMonteCarloAgent(MonteCarloAgent, ABC):

	def __init__(
			self,
			*args,
			discount=Config.AGENT_DISCOUNT_FACTOR,
			min_free_memory_percent=Config.AGENT_MIN_FREE_MEMORY,
			logical=Config.AGENT_LOGICAL_MCA,
			uct_exploration_weight=Config.AGENT_UCT_EXPLORE_WEIGHT,
			use_stm=Config.AGENT_STM,
			probability_correction=Config.AGENT_PROBABILITY_CORRECTION,
			min_probability=Config.AGENT_MIN_PROBABILITY,
			**kwargs
	):
		super(TraderMonteCarloAgent, self).__init__(
			*args,
			discount=discount,
			min_free_memory_percent=min_free_memory_percent,
			logical=logical,
			uct_exploration_weight=uct_exploration_weight,
			use_stm=use_stm,
			short_term_memory=AgentUtilsProvider.provide_trader_node_stm(),
			probability_correction=probability_correction,
			min_probability=min_probability,
			resource_manager=AgentUtilsProvider.provide_resource_manager(),
			**kwargs
		)

	def _get_state_node_instant_value(self, state_node: 'MonteCarloAgent.Node') -> float:
		return self._get_environment().get_reward(state_node.state) - self._get_environment().get_reward(
			state_node.parent.parent.state)
