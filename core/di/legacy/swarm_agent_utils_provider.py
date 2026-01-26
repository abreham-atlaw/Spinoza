from socketio import Client

from core import Config
from core.di import ServiceProvider
from core.utils.swarm.session_setup import SwarmQueenSetupManager, SwarmWorkerSetupManager
from lib.utils.cache.decorators import CacheDecorators


class SwarmAgentUtilsProvider:

	@staticmethod
	@CacheDecorators.singleton()
	def provide_socketio_client() -> Client:
		return Client(logger=True)

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
