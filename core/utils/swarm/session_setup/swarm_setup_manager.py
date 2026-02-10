import time
import typing
from abc import ABC, abstractmethod
from threading import Timer

import socketio.exceptions

from core.utils.swarm.session_setup.data.serializers import SessionSerializer
from lib.concurrency.swarm.sio_agent import SIOAgent
from lib.utils.logger import Logger


class SwarmSetupManager(SIOAgent, ABC):

	def __init__(
			self,
			server_url: str,
			*args,
			sleep_time: float = 1.0,
			reconnect_lag: float = 10.0,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self._server_url = server_url
		self._connect()
		self._session_serializer = SessionSerializer()
		self.__setup_complete = False
		self.__sleep_time = sleep_time
		self.__reconnected = False
		self.__reconnect_callbacks = []
		self.__reconnect_lag = reconnect_lag
		self._id = None

	def add_reconnect_callback(self, callback: typing.Callable[[], None]):
		self.__reconnect_callbacks.append(callback)

	def _connect(self, reconnect=False):
		try:
			Logger.info(f"[{self.__class__.__name__}] Connecting to {self._server_url}...")
			self._sio.connect(self._server_url)
			Logger.success(f"[{self.__class__.__name__}] Connected to {self._server_url}...")
		except socketio.exceptions.ConnectionError as ex:
			if not reconnect:
				raise ex
			if self._sio.connected:
				self._sio.reset()
				self._connect()

	def _handle_mca_start(self, data=None):
		self.__setup_complete = True
		self._id = data["id"]
		Logger.info(f"[{self.__class__.__name__}] Session setup with id: {self._id}")

	def __handle_mca_resume(self, data=None):
		Logger.success(f"[{self.__class__.__name__}] Session resumed")
		for callback in self.__reconnect_callbacks:
			callback()
		self.__reconnected = True

	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		return {
			"mca-start": self._handle_mca_start,
			"mca-resume": self.__handle_mca_resume
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

	@abstractmethod
	def _reconnect(self):
		pass

	def __reconnect(self):
		if self.__reconnected:
			Logger.success(f"[{self.__class__.__name__}] Reconnection Confirmed!")
			return
		Logger.info(f"[{self.__class__.__name__}] Reconnecting...")
		self._connect(reconnect=True)
		self._reconnect()

		Timer(self.__reconnect_lag, self.__reconnect).start()

	def reconnect(self):
		Logger.info(f"[{self.__class__.__name__}] Reconnecting after {self.__reconnect_lag}...")
		self.__reconnected = False
		timer = Timer(self.__reconnect_lag, self.__reconnect)
		timer.start()
