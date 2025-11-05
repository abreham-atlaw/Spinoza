from core import Config
from lib.rl.agent.mca.resource_manager import MCResourceManager, TimeMCResourceManager, DiskResourceManager
from lib.rl.agent.mca.stm import NodeMemoryMatcher, NodeShortTermMemory
from lib.utils.logger import Logger
from lib.utils.staterepository import StateRepository, AutoStateRepository, SectionalDictStateRepository, \
	PickleStateRepository


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
