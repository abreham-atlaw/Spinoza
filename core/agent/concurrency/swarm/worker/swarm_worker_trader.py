from core.agent.agents import TraderAgent
from core.di.legacy.swarm_agent_utils_provider import SwarmAgentUtilsProvider
from core.utils.swarm.session_setup import SwarmWorkerSetupManager
from lib.concurrency.swarm.worker import SwarmWorker, ReflexSwarmWorker


class SwarmWorkerTrader(ReflexSwarmWorker, SwarmWorker, TraderAgent):

	def __init__(self):
		super().__init__(
			resource_manager=SwarmAgentUtilsProvider.provide_mc_worker_resource_manager(),
			socket_client=SwarmAgentUtilsProvider.provide_socketio_client(),
			state_repository=SwarmAgentUtilsProvider.provide_mc_worker_state_repository(),
			node_serializer=SwarmAgentUtilsProvider.provide_node_serializer(),
		)
		self.__setup_manager = SwarmAgentUtilsProvider.provide_worker_setup_manager()
		self.__setup_manager.setup()
