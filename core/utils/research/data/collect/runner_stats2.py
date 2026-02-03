import typing
from dataclasses import dataclass, field
from datetime import datetime

from .runner_stats import RunnerStats


@dataclass
class RunnerStats2Session:
	session_timestamp: datetime
	simulated_timestamp: str
	profit: float
	model_loss: float
	timestep_pls: typing.List[float] = field(default_factory=lambda: [])

	@property
	def is_active(self) -> bool:
		return True in [
			value is None
			for value in [
				self.session_timestamp,
				self.simulated_timestamp,
				self.profit,
				self.model_loss
			]
		]

@dataclass
class RunnerStats2(RunnerStats):

	sessions: typing.List[RunnerStats2Session] = field(default_factory=lambda: [])
	aggregate_alpha: float = None

	@property
	def profits(self) -> typing.List[float]:
		return [session.profit for session in self.sessions]

	@property
	def session_timestamps(self) -> typing.List[datetime]:
		return [session.session_timestamp for session in self.sessions]

	@property
	def simulated_timestamps(self) -> typing.List[str]:
		return [session.simulated_timestamp for session in self.sessions]

	@property
	def session_model_losses(self) -> typing.List[float]:
		return [session.model_loss for session in self.sessions]

	def get_active_session(self):
		session  = self.sessions[-1]
		if not session.is_active:
			raise ValueError(f"No Active Session Found.")
		return session

	def create_session(self) -> RunnerStats2Session:
		self.sessions.append(RunnerStats2Session(
			*([None]*4)
		))
		return self.sessions[-1]

	def add_session_timestamp(self, timestamp: datetime):
		session = self.create_session()
		session.session_timestamp = timestamp

	def add_profit(self, profit: float):
		self.get_active_session().profit = profit

	def add_simulated_timestamp(self, timestamp: str):
		self.get_active_session().simulated_timestamp = timestamp

	def add_session_model_loss(self, loss: float):
		self.get_active_session().model_loss = loss

	@profits.setter
	def profits(self, value):
		pass

	@session_timestamps.setter
	def session_timestamps(self, value):
		pass

	@simulated_timestamps.setter
	def simulated_timestamps(self, value):
		pass

	@session_model_losses.setter
	def session_model_losses(self, value):
		pass