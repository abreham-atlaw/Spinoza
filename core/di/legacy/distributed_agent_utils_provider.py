import socketio

from core import Config
from lib.concurrency.mc.data.staterepository import DistributedStateRepository, SocketIOChannel
from lib.network.rest_interface import Serializer
from lib.rl.agent.mca.resource_manager import MCResourceManager, TimeMCResourceManager
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.logger import Logger
from lib.utils.staterepository import StateRepository


class DistributedAgentUtilsProvider:

	@staticmethod
	@CacheDecorators.singleton()
	def provide_worker_socket_io():
		return socketio.Client(logger=True)

	@staticmethod
	def provide_state_serializer() -> Serializer:
		from core.agent.concurrency.mc.data.serializer import TradeStateSerializer
		return TradeStateSerializer()

	@staticmethod
	def provide_node_serializer() -> Serializer:
		from core.agent.concurrency.mc.data.serializer import TraderNodeSerializer
		return TraderNodeSerializer()

	@staticmethod
	def provide_mc_worker_state_repository() -> StateRepository:
		return DistributedStateRepository(
				SocketIOChannel(
					DistributedAgentUtilsProvider.provide_worker_socket_io(),
				),
				DistributedAgentUtilsProvider.provide_state_serializer(),
				is_server=False
			)

	@staticmethod
	def provide_mc_worker_resource_manager() -> MCResourceManager:
		manager = TimeMCResourceManager(
			step_time=Config.MC_WORKER_STEP_TIME
		)
		Logger.info(f"Using MC Worker Resource Manager: {manager.__class__.__name__}")
		return manager
