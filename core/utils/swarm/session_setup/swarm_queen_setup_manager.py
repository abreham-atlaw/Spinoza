import os.path
import time
import typing

from socketio import Client

from core import Config
from core.utils.swarm.session_setup import SwarmSetupManager
from core.utils.swarm.session_setup.data.models import Session
from core.utils.swarm.session_setup.data.serializers import SessionSerializer
from lib.concurrency.swarm.sio_agent import SIOAgent
from lib.utils.logger import Logger


class SwarmQueenSetupManager(SwarmSetupManager):

	@staticmethod
	def __construct_session() -> Session:
		return Session(
			branch=Config.RunnerStatsBranches.default,
			model=os.path.basename(Config.CORE_MODEL_CONFIG.path),
			model_temperature=Config.AGENT_MODEL_TEMPERATURE,
			model_alpha=Config.AGENT_MODEL_AGGREGATION_ALPHA
		)

	def _setup(self):
		session = self.__construct_session()
		self._sio.emit(
			"create-session",
			data=self._session_serializer.serialize(session)
		)
