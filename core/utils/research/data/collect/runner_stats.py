import typing

from dataclasses import dataclass, field
from datetime import datetime

from core import Config


@dataclass
class RunnerStats:

	id: str
	model_name: str
	session_timestamps: typing.List[datetime]
	profits: typing.List[float] = field(default_factory=lambda: [])
	duration: float = 0.0
	model_losses_map: typing.Dict[str, typing.Tuple[float, ...]] = field(default_factory=lambda: {})
	temperature: float = 1.0

	@property
	def profit(self) -> float:
		return sum(self.profits)

	@property
	def model_losses(self) -> typing.Tuple[float, ...]:
		return self.get_model_losses()

	@model_losses.setter
	def model_losses(self, value: typing.Tuple[float, ...]):
		self.model_losses_map[Config.RunnerStatsLossesBranches.default] = value

	def get_model_losses(self, key: str = None) -> typing.Tuple[float,...]:
		if key is None:
			key = Config.RunnerStatsLossesBranches.default
		return self.model_losses_map.get(key, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

	def add_profit(self, profit: float):
		self.profits.append(profit)

	def add_duration(self, duration: float):
		self.duration += duration

	def add_session_timestamp(self, timestamp: datetime):
		self.session_timestamps.append(timestamp)
