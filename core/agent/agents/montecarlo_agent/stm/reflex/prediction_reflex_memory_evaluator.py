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
			bounds: typing.List[float]
	):
		super().__init__()
		self.__state_predictor = state_predictor
		self.__bounds = DataPrepUtils.apply_bound_epsilon(bounds)
		Logger.info(f"Initialized {type(self).__name__} with state_predictor: {state_predictor}")

	@CacheDecorators.cached_method(size=10000)
	def __predict_state(self, state0: TradeState) -> np.ndarray:
		y = np.concatenate([
			self.__state_predictor.predict([state0], [None], instrument=instrument)
			for instrument in state0.get_market_state().get_tradable_pairs()
		], axis=1)[0]

		return np.sum(self.__bounds * y, axis=-1)

	def _evaluate_market_state(self, state0: TradeState, state1: TradeState) -> float:
		y0, y1 = self.__predict_state(state0), self.__predict_state(state1)
		return np.sum(np.abs(y0 - y1))

	@staticmethod
	def _evaluate_agent_state(state0: AgentState, state1: AgentState) -> float:
		return 0
