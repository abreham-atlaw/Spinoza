import os.path

from core import Config
from core.utils.swarm.session_setup import SwarmSetupManager
from core.utils.swarm.session_setup.data.models import Session


class SwarmQueenSetupManager(SwarmSetupManager):

	def __init__(
			self,
			*args,
			**kwargs,
	):
		super().__init__(*args, **kwargs)

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

	def _reconnect(self):
		self._sio.emit(
			"queen-reconnect",
			data={
				"id": self._id
			}
		)


