from .action_choice_trader.action_choice_trader import ActionChoiceTrader
from .cra import CumulativeRewardTraderAgent
from .direct_probability_distribution_agent import DirectProbabilityDistributionAgent
from .drmca import TraderDeepReinforcementMonteCarloAgent
from .montecarlo_agent import TraderMonteCarloAgent, ReflexAgent
from core import Config


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
