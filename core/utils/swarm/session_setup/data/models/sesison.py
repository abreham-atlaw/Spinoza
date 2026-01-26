import typing
from dataclasses import dataclass


@dataclass
class Session:
	branch: str
	model: str
	model_temperature: float
	model_alpha: typing.Optional[float]
