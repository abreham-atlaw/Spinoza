import typing

import numpy as np

from core.agent.utils.state_predictor import BasicStatePredictor
from core.environment.trade_state import TradeState, AgentState
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.logger import Logger
from .trader_reflex_memory_evaluator import TraderReflexMemoryEvaluator


class PredictionReflexMemoryEvaluator(TraderReflexMemoryEvaluator):

	def __init__(
			self,
			state_predictor: BasicStatePredictor,
			bounds: typing.List[float],
			effective_channels: typing.List[int] = None,
			y_extra_len: int = 1,
			log_returns: bool = False,
			focused_instrument_simulation: bool = False
	):
		super().__init__()
		self.__state_predictor = state_predictor
		self.__bounds = DataPrepUtils.apply_bound_epsilon(bounds)
		self.__effective_channels = effective_channels
		self.__y_extra_len = y_extra_len
		self.__log_returns = log_returns
		self.__focused_instrument_simulation = focused_instrument_simulation
		Logger.info(
			f"Initialized {type(self).__name__} with state_predictor: {state_predictor}, "
			f"effective_channels: {effective_channels}, log_returns={log_returns}, "
			f"focused_instrument_simulation: {focused_instrument_simulation}"
		)

	def __predict_state_instrument(self, state0: TradeState, instrument: typing.Tuple[str, str]) -> np.ndarray:
		y = self.__state_predictor.predict([state0], [None], instrument=instrument)[0]
		if self.__y_extra_len > 0:
			y = y[..., :-self.__y_extra_len]

		bounds = self.__bounds
		if self.__log_returns:
			bounds = np.exp(bounds)

		y = np.sum(bounds * y, axis=-1)
		if self.__effective_channels is not None:
			y = y[self.__effective_channels]
		return y


	def __predict_state(self, state0: TradeState, instruments: typing.List[typing.Tuple[str, str]]) -> np.ndarray:
		return np.concatenate([
			self.__predict_state_instrument(state0, instrument=instrument)
			for instrument in instruments
		])

	def __select_instruments(self, state0: TradeState, state1: TradeState) -> typing.List[typing.Tuple[str, str]]:
		if not self.__focused_instrument_simulation:
			return state0.get_market_state().get_tradable_pairs()
		instruments = list(set([
			state.simulated_instrument
			for state in [state0, state1]
			if state.simulated_instrument is not None
		]))
		if len(instruments) > 0:
			return instruments
		return state0.get_market_state().get_tradable_pairs()

	def _evaluate_market_state(self, state0: TradeState, state1: TradeState) -> float:
		instruments = self.__select_instruments(state0, state1)

		y0, y1 = self.__predict_state(state0, instruments), self.__predict_state(state1, instruments)
		return np.sum(np.abs(y0 - y1))

	@staticmethod
	def _evaluate_agent_state(state0: AgentState, state1: AgentState) -> float:
		return 0
