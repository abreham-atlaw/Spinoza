import typing

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
		self.__setup_manager.add_reconnect_callback(self.__handle_reconnect)
		self.__setup_manager.setup()
		self._map_events_map()
		self.__handling_disconnect = False

	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		events_map = super()._map_events()
		events_map.update({
			"disconnect": self._handle_disconnect
		})
		return events_map

	def __handle_reconnect(self):
		self._emit_select()
		self.__handling_disconnect = False

	def _handle_disconnect(self):
		super()._handle_disconnect()
		if self.__handling_disconnect:
			return
		self.__handling_disconnect = True
		self.__setup_manager.reconnect()
