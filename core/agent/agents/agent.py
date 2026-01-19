from .action_choice_trader.action_choice_trader import ActionChoiceTrader
from .cra import CumulativeRewardTraderAgent
from .drmca import TraderDeepReinforcementMonteCarloAgent
from .montecarlo_agent import TraderMonteCarloAgent, ReflexAgent
from core import Config
from .direct_probability_distribution_agent import DirectProbabilityDistributionAgent



bases = (
    (ReflexAgent,) if Config.AGENT_MCA_USE_REFLEX else ()
) + (
	DirectProbabilityDistributionAgent,
    CumulativeRewardTraderAgent,
    TraderDeepReinforcementMonteCarloAgent,
    TraderMonteCarloAgent,
    ActionChoiceTrader,
)


class TraderAgent(*bases):
    pass
