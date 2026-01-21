import socketio

from core.di import AgentUtilsProvider
from core.di.legacy import DistributedAgentUtilsProvider
from lib.concurrency.mc.worker import MonteCarloWorkerAgent, ReflexMonteCarloWorker
from lib.network.rest_interface import Serializer
from core.agent.agents import TraderMonteCarloAgent, TraderAgent
from core.agent.concurrency.mc.data.serializer import TraderNodeSerializer, TradeStateSerializer
from core import Config


class TraderMonteCarloWorkerAgent(ReflexMonteCarloWorker, TraderAgent, MonteCarloWorkerAgent):

	def __init__(self):
		super().__init__(
			server_url=Config.MC_SERVER_URL,
			resource_manager=DistributedAgentUtilsProvider.provide_mc_worker_resource_manager(),
			state_repository=DistributedAgentUtilsProvider.provide_mc_worker_state_repository(),
		)

	def _init_state_serializer(self) -> Serializer:
		return DistributedAgentUtilsProvider.provide_state_serializer()

	def _init_graph_serializer(self) -> Serializer:
		return DistributedAgentUtilsProvider.provide_node_serializer()

	def _init_socket_io(self) -> socketio.Client:
		return DistributedAgentUtilsProvider.provide_worker_socket_io()
