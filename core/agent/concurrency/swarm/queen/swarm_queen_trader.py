from core.agent.agents import TraderAgent
from core.di.legacy.swarm_agent_utils_provider import SwarmAgentUtilsProvider
from lib.concurrency.swarm.queen.swarm_queen import SwarmQueen


class SwarmQueenTrader(SwarmQueen, TraderAgent):

	def __init__(self):
		super().__init__(
			socket_client=SwarmAgentUtilsProvider.provide_socketio_client(),
			state_repository=SwarmAgentUtilsProvider.provide_mc_worker_state_repository(),
			node_serializer=SwarmAgentUtilsProvider.provide_node_serializer()
		)
		self.__setup_manager = SwarmAgentUtilsProvider.provide_queen_setup_manager()
		self.__setup_manager.setup()
