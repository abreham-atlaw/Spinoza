import typing

import numpy as np

from core.agent.utils.state_predictor import StatePredictor, BasicStatePredictor
from core.environment.trade_state import MarketState, TradeState, AgentState
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.logger import Logger
from .trader_reflex_memory_evaluator import TraderReflexMemoryEvaluator


class PredictionReflexMemoryEvaluator(TraderReflexMemoryEvaluator):

	def __init__(
			self,
			state_predictor: BasicStatePredictor,
			bounds: typing.List[float],
			effective_channels: typing.List[int] = None
	):
		super().__init__()
		self.__state_predictor = state_predictor
		self.__bounds = DataPrepUtils.apply_bound_epsilon(bounds)
		self.__effective_channels = effective_channels
		Logger.info(f"Initialized {type(self).__name__} with state_predictor: {state_predictor}, effective_channels: {effective_channels}")

	def __predict_state_instrument(self, state0: TradeState, instrument: typing.Tuple[str, str]) -> np.ndarray:
		y = self.__state_predictor.predict([state0], [None], instrument=instrument)[0]
		y = np.sum(self.__bounds * y, axis=-1)
		if self.__effective_channels is not None:
			y = y[self.__effective_channels]
		return y

	@CacheDecorators.cached_method(size=10000)
	def __predict_state(self, state0: TradeState) -> np.ndarray:
		return np.concatenate([
			self.__predict_state_instrument(state0, instrument=instrument)
			for instrument in state0.get_market_state().get_tradable_pairs()
		])

	def _evaluate_market_state(self, state0: TradeState, state1: TradeState) -> float:
		y0, y1 = self.__predict_state(state0), self.__predict_state(state1)
		return np.sum(np.abs(y0 - y1))

	@staticmethod
	def _evaluate_agent_state(state0: AgentState, state1: AgentState) -> float:
		return 0
