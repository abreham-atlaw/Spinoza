import os.path
import time
import typing
from threading import Timer

from core import Config
from core.utils.swarm.session_setup import SwarmSetupManager
from core.utils.swarm.session_setup.data.models import Session
from core.utils.swarm.session_setup.data.serializers import SessionSerializer
from lib.concurrency.swarm.sio_agent import SIOAgent
from lib.utils.logger import Logger


class SwarmQueenSetupManager(SwarmSetupManager):

	def __init__(
			self,
			*args,
			reconnect_lag: float = 5.0,
			**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.__id = None
		self.__reconnect_callbacks = []
		self.__reconnect_lag = reconnect_lag

	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		event_map = super()._map_events()
		event_map.update({
			"mca-resume": self.__handle_mca_resume
		})
		return event_map

	def add_reconnect_callback(self, callback: typing.Callable[[], None]):
		self.__reconnect_callbacks.append(callback)

	@staticmethod
	def __construct_session() -> Session:
		return Session(
			branch=Config.RunnerStatsBranches.default,
			model=os.path.basename(Config.CORE_MODEL_CONFIG.path),
			model_temperature=Config.AGENT_MODEL_TEMPERATURE,
			model_alpha=Config.AGENT_MODEL_AGGREGATION_ALPHA
		)

	def _handle_mca_start(self, data=None):
		super()._handle_mca_start()
		self.__id = data["id"]
		Logger.info(f"[SwarmQueenSetupManager] Session setup with id: {self.__id}")

	def __handle_mca_resume(self, data=None):
		Logger.success(f"[SwarmQueenSetupManager] Session resumed")
		for callback in self.__reconnect_callbacks:
			callback()

	def _setup(self):
		session = self.__construct_session()
		self._sio.emit(
			"create-session",
			data=self._session_serializer.serialize(session)
		)

	def __reconnect(self):
		Logger.info(f"[SwarmQueenSetupManager] Reconnecting...")
		self._connect()
		self._sio.emit(
			"queen-reconnect",
			data={
				"id": self.__id
			}
		)

	def reconnect(self):
		Logger.info(f"[SwarmQueenSetupManager] Reconnecting after {self.__reconnect_lag}...")
		timer = Timer(self.__reconnect_lag, self.__reconnect)
		timer.start()

