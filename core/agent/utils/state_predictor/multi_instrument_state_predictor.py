import typing

import numpy as np

from core.environment.trade_state import TradeState
from .state_predictor import StatePredictor
from ...action import Action


class MultiInstrumentPredictor(StatePredictor):

	def _prepare_input(
			self,
			states: typing.List[TradeState],
			actions: typing.List[Action],
			*args, **kwargs
	) -> np.ndarray:

		X = np.stack([
			np.concatenate([
				state.get_market_state().get_channels_state(*instrument)
				for instrument in state.get_market_state().get_tradable_pairs()
			], axis=0)
			for state in states
		])

		return X

	def _prepare_output(self, y: np.ndarray, states: typing.List[TradeState], actions: typing.List[Action], *args, **kwargs) -> np.ndarray:

		instrument = kwargs.get("instrument")

		channels = states[0].get_market_state().channels
		y = np.stack([
			y[:, i*channels: (i+1)*channels]
			for i in range(y.shape[1] // channels)
		], axis=1)

		if instrument is not None:
			idx = states[0].get_market_state().get_tradable_pairs().index(instrument)
			return y[:, idx]

		return y
