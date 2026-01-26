import os.path
import typing

from core import Config
from core.di import ServiceProvider
from core.utils.swarm.session_setup import SwarmSetupManager
from core.utils.swarm.session_setup.data.models import Session
from lib.concurrency.swarm.sio_agent import SIOAgent
from lib.utils.file_storage import FileStorage
from lib.utils.logger import Logger


class SwarmWorkerSetupManager(SwarmSetupManager):

	def __init__(
			self,
			*args,
			fs: FileStorage,
			**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.__fs = fs

	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		event_map = super()._map_events()
		event_map.update({
			"setup": self.__handle_setup
		})
		return event_map

	def __setup_session(self, session: Session):
		Logger.info(f"Downloading model: {session.model}")
		Config.CORE_MODEL_CONFIG.path = os.path.abspath(session.model)
		self.__fs.download(session.model, Config.CORE_MODEL_CONFIG.path)
		Config.AGENT_MODEL_TEMPERATURE = session.model_temperature
		Config.AGENT_MODEL_AGGREGATION_ALPHA = session.model_alpha
		Config.AGENT_MODEL_USE_AGGREGATION = session.model_alpha is not None

	def __handle_setup(self, session_data):
		Logger.info(f"Received setup for session {session_data}")
		session = self._session_serializer.deserialize(session_data)
		self.__setup_session(session)

		self._sio.emit(
			"setup-complete",
		)
		Logger.success(f"Swarm Session Setup Complete")

	def __register_worker(self):
		Logger.info(f"Registering worker...")
		self._sio.emit(
			"register-worker",
			data={
				"branch": Config.RunnerStatsBranches.default
			}
		)

	def _setup(self):
		self.__register_worker()
