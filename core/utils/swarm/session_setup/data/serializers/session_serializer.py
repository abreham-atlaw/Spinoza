from typing import Dict

from core.utils.swarm.session_setup.data.models import Session
from lib.network.rest_interface import Serializer
from lib.network.rest_interface.serializers import T


class SessionSerializer(Serializer[Session]):

	def __init__(self):
		super().__init__(output_class=Session)

	def serialize(self, data: Session) -> Dict:
		return {
			"branch": data.branch,
			"model": data.model,
			"model_temperature": data.model_temperature,
			"model_alpha": data.model_alpha
		}

	def deserialize(self, json_: Dict) -> Session:
		return Session(
			branch=json_["branch"],
			model=json_["model"],
			model_temperature=json_["model_temperature"],
			model_alpha=json_["model_alpha"]
		)
