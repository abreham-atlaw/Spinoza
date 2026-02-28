import typing

import numpy as np

from core.environment.trade_state import TradeState
from lib.rl.agent.utils.state_predictor import StatePredictor
from core.agent.action import Action


class BasicStatePredictor(StatePredictor):

	def __init__(self, *args, extra_len: int=0, y_extra_len: int=1, **kwargs):
		super().__init__(*args, **kwargs)
		self.__extra_len = extra_len
		self.__y_extra_len = y_extra_len

	def prepare_input(
			self,
			states: typing.List[TradeState],
			actions: typing.List[Action],
			*args, **kwargs
	) -> np.ndarray:
		instrument = kwargs.get("instrument")


		if instrument is None:
			instrument = states[0].get_market_state().get_tradable_pairs()[0]



		seq = np.stack([
			state.get_market_state().get_channels_state(*instrument)
			for state in states
		], axis=0)

		X = np.concatenate(
			(
				seq,
				np.zeros(seq.shape[:-1]+(self.__extra_len,)),
			),
			axis=-1
		)

		return X

	def _prepare_output(self, y: np.ndarray, states: typing.List[TradeState], actions: typing.List[Action], *args, **kwargs) -> np.ndarray:
		return y
