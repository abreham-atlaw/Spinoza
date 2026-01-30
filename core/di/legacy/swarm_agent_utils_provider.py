from socketio import Client

from core import Config
from core.di import ServiceProvider, DistributedAgentUtilsProvider
from core.utils.swarm.session_setup import SwarmQueenSetupManager, SwarmWorkerSetupManager
from lib.concurrency.mc.data.staterepository import DistributedStateRepository, SocketIOChannel
from lib.network.rest_interface import Serializer
from lib.rl.agent.mca.resource_manager import MCResourceManager, TimeMCResourceManager
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.logger import Logger
from lib.utils.staterepository import StateRepository


class SwarmAgentUtilsProvider:

	@staticmethod
	@CacheDecorators.singleton()
	def provide_socketio_client() -> Client:
		return Client(logger=Config.SWARM_SOCKETIO_LOGGING)

	@staticmethod
	def provide_queen_setup_manager() -> SwarmQueenSetupManager:
		return SwarmQueenSetupManager(
			socket_client=SwarmAgentUtilsProvider.provide_socketio_client(),
			server_url=Config.SWARM_HUB_HOST,
		)

	@staticmethod
	def provide_worker_setup_manager() -> SwarmWorkerSetupManager:
		return SwarmWorkerSetupManager(
			socket_client=SwarmAgentUtilsProvider.provide_socketio_client(),
			server_url=Config.SWARM_HUB_HOST,
			fs=ServiceProvider.provide_file_storage(Config.OANDA_SIM_MODEL_IN_PATH),
		)

	@staticmethod
	def provide_mc_worker_state_repository() -> StateRepository:
		return DistributedStateRepository(
			SocketIOChannel(
				SwarmAgentUtilsProvider.provide_socketio_client(),
			),
			DistributedAgentUtilsProvider.provide_state_serializer(),
			is_server=False
		)

	@staticmethod
	def provide_mc_worker_resource_manager() -> MCResourceManager:
		manager = TimeMCResourceManager(
			step_time=Config.SWARM_WORKER_STEP_TIME
		)
		Logger.info(f"Using MC Worker Resource Manager: {manager.__class__.__name__}")
		return manager

	@staticmethod
	def provide_node_serializer() -> Serializer:
		from core.agent.concurrency.mc.data.serializer import TraderNodeSerializer
		return TraderNodeSerializer()