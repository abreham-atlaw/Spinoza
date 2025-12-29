from .action_choice_trader.action_choice_trader import ActionChoiceTrader
from .cra import CumulativeRewardTraderAgent
from .drmca import TraderDeepReinforcementMonteCarloAgent
from .montecarlo_agent import TraderMonteCarloAgent
from .montecarlo_agent import ReflexAgent
from .direct_probability_distribution_agent import DirectProbabilityDistributionAgent


class TraderAgent(
	ReflexAgent,
	DirectProbabilityDistributionAgent,
	CumulativeRewardTraderAgent,
	TraderDeepReinforcementMonteCarloAgent,
	TraderMonteCarloAgent,
	ActionChoiceTrader,
):
	pass
