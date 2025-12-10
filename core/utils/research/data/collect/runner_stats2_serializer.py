from typing import Dict

from lib.network.rest_interface import Serializer
from .runner_stats import RunnerStats
from .runner_stats_serializer import RunnerStatsSerializer
from .runner_stats2 import RunnerStats2, RunnerStats2Session

class SessionSerializer(Serializer):

	def __init__(self):
		super().__init__(RunnerStats2Session)

	def serialize(self, data: RunnerStats2Session):
		json_ = data.__dict__.copy()
		return json_

	def deserialize(self, json_: Dict) -> RunnerStats2Session:
		return RunnerStats2Session(**json_)


class RunnerStats2Serializer(Serializer):

	__omitted_keys = ["_id", "session_timestamps", "simulated_timestamps", "profits", "session_model_losses", "real_profits", "branch"]

	def __init__(self):
		super().__init__(RunnerStats2Session)
		self.__session_serializer = SessionSerializer()
		self.__v1_serializer = RunnerStatsSerializer()

	def serialize(self, data: RunnerStats2) -> Dict:
		json_ = data.__dict__.copy()
		for key in self.__omitted_keys:
			if key in json_:
				json_.pop(key)

		json_["sessions"] = self.__session_serializer.serialize_many(data.sessions)

		return json_

	@staticmethod
	def __is_v1(json_: Dict) -> bool:
		return "sessions" not in json_

	def __from_v1(self, stat: RunnerStats) -> RunnerStats2:
		return RunnerStats2(
			id=stat.id,
			model_name=stat.model_name,
			duration=stat.duration,
			temperature=stat.temperature,
			model_losses_map=stat.model_losses_map,
			sessions=[
				RunnerStats2Session(
					session_timestamp=stat.session_timestamps[i],
					simulated_timestamp=stat.simulated_timestamps[i] if len(stat.simulated_timestamps) > 0 else None,
					profit=stat.profits[i],
					model_loss=stat.session_model_losses[i]
				)
				for i in range(len(stat.session_timestamps))
			]
		)

	def __process_v1(self, json_: Dict):
		v1_stat = self.__v1_serializer.deserialize(json_)
		return self.__from_v1(v1_stat)

	def deserialize(self, json_: Dict) -> RunnerStats2:
		if self.__is_v1(json_):
			return self.__process_v1(json_)

		for key in self.__omitted_keys:
			if key in json_:
				json_.pop(key)

		json_["sessions"] = self.__session_serializer.deserialize_many(json_["sessions"])

		return RunnerStats2(**json_)
