import typing
from abc import ABC
from copy import deepcopy

import numpy as np

import chess
from torch import nn

from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from lib.rl.agent.dra.dra import DeepReinforcementAgent
from lib.rl.agent.dta import Model, TorchModel
from lib.utils.torch_utils.model_handler import ModelHandler
from test.lib.rl.environment.environments.chess import ChessState


class ChessDRCNNModel(LinearModel):

	def __init__(self, *args, **kwargs):
		super().__init__(
			block_size=896,
			vocab_size=1,
			*args,
			**kwargs
		)


class ChessDeepReinforcementAgent(DeepReinforcementAgent, ABC):

	def _init_predictor(self) -> Model:
		return TorchModel(ChessDRCNNModel(
			layer_sizes=[1024, 2048],
			hidden_activation=nn.ReLU(),
			norm=[True, False, False]
		))

	@staticmethod
	def __encode_board(board):
		encoding = np.zeros((14, 8, 8), dtype=np.float32)
		for i in range(64):
			piece = board.piece_at(i)
			if piece is not None:
				encoding[piece.piece_type + int(piece.color) * 6][i // 8][i % 8] = 1
		encoding[13] = 1 if board.turn == chess.BLACK else 0
		return encoding.reshape(-1)

	def _prepare_dra_input(self, state: ChessState, action: chess.Move) -> np.ndarray:
		mid_board = deepcopy(state.get_board())
		mid_board.push(action)
		return self.__encode_board(mid_board)

	def _prepare_dra_output(self, state: typing.Any, action: typing.Any, output: np.ndarray) -> float:
		return float(output.flatten()[0])

	def _prepare_dra_train_output(self, state: typing.Any, action: typing.Any, value: float) -> np.ndarray:
		return np.array(value)
