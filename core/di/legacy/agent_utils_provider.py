from core import Config
from core.agent.utils.state_transition_sampler import StateTransitionSampler, BasicStateTransitionSampler
from core.utils.research.model.model.utils import AggregateModel, WrappedModel, TransitionOnlyModel, \
	TemperatureScalingModel
from core.utils.research.model.model.utils.cached_model import CachedModel
from lib.rl.agent.dta import TorchModel
from lib.rl.agent.mca.resource_manager import MCResourceManager, TimeMCResourceManager, DiskResourceManager
from lib.rl.agent.mca.stm import NodeMemoryMatcher, NodeShortTermMemory
from lib.utils.logger import Logger
from lib.utils.staterepository import StateRepository, AutoStateRepository, SectionalDictStateRepository, \
	PickleStateRepository
from lib.utils.stm import StochasticMemoryEvaluator, StochasticShortTermMemory
from lib.utils.torch_utils.model_handler import ModelHandler


class AgentUtilsProvider:

	@staticmethod
	def provide_in_memory_state_repository() -> StateRepository:
		return SectionalDictStateRepository(2, 15)

	@staticmethod
	def provide_disk_state_repository() -> StateRepository:
		return PickleStateRepository(
			path=Config.AGENT_FILESYSTEM_STATE_REPOSITORY_PATH
		)

	@staticmethod
	def provide_state_repository() -> StateRepository:
		if Config.AGENT_USE_AUTO_STATE_REPOSITORY:
			Logger.info("Using auto state repository...")
			return AutoStateRepository(
				Config.AGENT_AUTO_STATE_REPOSITORY_MEMORY_SIZE,
				in_memory_repository=AgentUtilsProvider.provide_in_memory_state_repository(),
				disk_repository=AgentUtilsProvider.provide_disk_state_repository()
			)
		Logger.info("Using in-memory state repository...")
		return AgentUtilsProvider.provide_in_memory_state_repository()

	@staticmethod
	def provide_disk_resource_manager() -> DiskResourceManager:
		return DiskResourceManager(
			min_remaining_space=Config.AGENT_MIN_DISK_SPACE,
			min_abs_remaining_space=Config.AGENT_MIN_ABS_DISK_SPACE,
			path=Config.AGENT_FILESYSTEM_STATE_REPOSITORY_PATH
		)

	@staticmethod
	def provide_resource_manager() -> MCResourceManager:
		from core.agent.agents.montecarlo_agent.trader_resource_manager import TraderMCResourceManager
		from .environment_utils_provider import EnvironmentUtilsProvider

		if Config.AGENT_USE_CUSTOM_RESOURCE_MANAGER:
			manager = TraderMCResourceManager(
				trader=EnvironmentUtilsProvider.provide_trader(),
				granularity=Config.MARKET_STATE_GRANULARITY,
				instrument=Config.AGENT_STATIC_INSTRUMENTS[0],
				delta_multiplier=Config.OANDA_SIM_DELTA_MULTIPLIER,
				disk_resource_manager=AgentUtilsProvider.provide_disk_resource_manager()
			)
		else:
			manager = TimeMCResourceManager(
				step_time=Config.AGENT_STEP_TIME
			)
		Logger.info(f"Using Resource Manager: {manager.__class__.__name__}")
		return manager

	@staticmethod
	def provide_agent_state_memory_matcher() -> 'AgentStateMemoryMatcher':
		from core.agent.agents.montecarlo_agent.stm.asmm import BasicAgentStateMemoryMatcher
		return BasicAgentStateMemoryMatcher()

	@staticmethod
	def provide_market_state_memory_matcher() -> 'MarketStateMemoryMatcher':
		from core.agent.agents.montecarlo_agent.stm.msmm import BoundMarketStateMemoryMatcher
		return BoundMarketStateMemoryMatcher(bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND)

	@staticmethod
	def provide_trade_state_memory_matcher() -> 'TradeStateMemoryMatcher':
		from core.agent.agents.montecarlo_agent.stm.tsmm import TradeStateMemoryMatcher
		return TradeStateMemoryMatcher(
			agent_state_matcher=AgentUtilsProvider.provide_agent_state_memory_matcher(),
			market_state_matcher=AgentUtilsProvider.provide_market_state_memory_matcher()
		)

	@staticmethod
	def provide_trade_node_memory_matcher(repository=None) -> NodeMemoryMatcher:
		matcher = NodeMemoryMatcher(
			repository=repository,
			state_matcher=AgentUtilsProvider.provide_trade_state_memory_matcher()
		)
		Logger.info(f"Using Trade Node Memory Matcher: {matcher.__class__.__name__}")
		return matcher

	@staticmethod
	def provide_trader_node_stm() -> NodeShortTermMemory:
		memory = NodeShortTermMemory(
			size=Config.AGENT_STM_SIZE,
			matcher=AgentUtilsProvider.provide_trade_node_memory_matcher()
		)
		Logger.info(f"Using Trade Node STM: {memory.__class__.__name__}")
		return memory

	@staticmethod
	def provide_core_torch_model() -> TorchModel:
		model = TemperatureScalingModel(
			model=ModelHandler.load(Config.CORE_MODEL_CONFIG.path),
			temperature=Config.AGENT_MODEL_TEMPERATURE
		)
		print(f"Using Temperature: {Config.AGENT_MODEL_TEMPERATURE}")
		if Config.AGENT_MODEL_USE_CACHED_MODEL:
			model = CachedModel(
				model=model,
			)
		if Config.AGENT_MODEL_USE_TRANSITION_ONLY:
			model = TransitionOnlyModel(
				model=model,
				extra_len=Config.AGENT_MODEL_EXTRA_LEN
			)
		model = WrappedModel(
			model,
			seq_len=Config.MARKET_STATE_MEMORY,
			window_size=Config.AGENT_MA_WINDOW_SIZE,
			use_ma=Config.AGENT_USE_SMOOTHING,
		)

		if Config.AGENT_MODEL_USE_AGGREGATION:
			model = AggregateModel(
				model=model,
				a=Config.AGENT_MODEL_AGGREGATION_ALPHA,
				bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
			)
		return TorchModel(
			model
		)

	@staticmethod
	def provide_state_predictor() -> 'StatePredictor':
		from core.agent.utils.state_predictor import BasicStatePredictor, LegacyStatePredictor, MultiInstrumentPredictor

		if Config.AGENT_USE_MULTI_INSTRUMENT_MODEL:
			Logger.info(f"Using MultiInstrumentPredictor...")
			return MultiInstrumentPredictor(
				model=AgentUtilsProvider.provide_core_torch_model(),
			)

		if Config.MARKET_STATE_USE_MULTI_CHANNELS:
			Logger.info(f"Using BasicStateProvider...")
			return BasicStatePredictor(
				model=AgentUtilsProvider.provide_core_torch_model(),
				extra_len=Config.AGENT_MODEL_EXTRA_LEN
			)


		Logger.info(f"Using LegacyStatePredictor...")
		return LegacyStatePredictor(
			model=AgentUtilsProvider.provide_core_torch_model(),
			extra_len=Config.AGENT_MODEL_EXTRA_LEN
		)

	@staticmethod
	def provide_reflex_memory_evaluator() -> StochasticMemoryEvaluator:
		from core.agent.agents.montecarlo_agent.stm.reflex import PredictionReflexMemoryEvaluator
		return PredictionReflexMemoryEvaluator(
			state_predictor=AgentUtilsProvider.provide_state_predictor(),
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			effective_channels=Config.AGENT_PREDICTION_REFLEX_EVALUATOR_EFFECTIVE_CHANNELS
		)

	@staticmethod
	def provide_reflex_stm() -> StochasticShortTermMemory:
		return StochasticShortTermMemory(
			size=Config.AGENT_REFLEX_STM_SIZE,
			evaluator=AgentUtilsProvider.provide_reflex_memory_evaluator(),
		)

	@staticmethod
	def provide_state_transition_sampler() -> StateTransitionSampler:
		return BasicStateTransitionSampler(
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			channels=Config.MARKET_STATE_CHANNELS,
			simulated_channels=Config.MARKET_STATE_SIMULATED_CHANNELS,
		)
