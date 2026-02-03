import time
import typing
from abc import ABC, abstractmethod

import socketio.exceptions

from core.utils.swarm.session_setup.data.serializers import SessionSerializer
from lib.concurrency.swarm.sio_agent import SIOAgent
from lib.utils.logger import Logger


class SwarmSetupManager(SIOAgent, ABC):

	def __init__(self, server_url: str, *args, sleep_time: float = 1.0, **kwargs):
		super().__init__(*args, **kwargs)
		self._server_url = server_url
		self._connect()
		self._session_serializer = SessionSerializer()
		self.__setup_complete = False
		self.__sleep_time = sleep_time

	def _connect(self, reconnect=False):
		try:
			Logger.info(f"[{self.__class__.__name__}] Connecting to {self._server_url}...")
			self._sio.connect(self._server_url)
		except socketio.exceptions.ConnectionError as ex:
			if not reconnect:
				raise ex
			if self._sio.connected:
				self._sio.disconnect()
				self._connect()

	def _handle_mca_start(self, data=None):
		self.__setup_complete = True

	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		return {
			"mca-start": self._handle_mca_start
		}

	def __wait_acknowledgement(self) -> None:
		Logger.info(f"[{self.__class__.__name__}] Waiting for Acknowledgement...")
		while not self.__setup_complete:
			time.sleep(self.__sleep_time)
		Logger.success(f"[{self.__class__.__name__}] Swarm Queen Setup Complete...")

	@abstractmethod
	def _setup(self):
		pass

	def setup(self):
		Logger.info(f"[{self.__class__.__name__}] Setting up Session...")
		self._setup()
		self.__wait_acknowledgement()