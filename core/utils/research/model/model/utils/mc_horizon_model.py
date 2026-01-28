import typing

import torch

from lib.utils.logger import Logger
from .horizon_model import HorizonModel


class MCHorizonModel(HorizonModel):

	def __init__(self, *args, y_channel_map: typing.Tuple[int, ...] = None, **kwargs):
		super().__init__(*args, **kwargs)
		Logger.info(f"Initializing MCHorizonModel...")
		self.args.update({
			"y_channel_map": y_channel_map
		})
		if y_channel_map is None:
			y_channel_map = (0,)
		self.y_channel_map = y_channel_map

	def _retrieve_recent_close(self, x: torch.Tensor):
		return  x[..., self.y_channel_map,  -(self.X_extra_len + 1)]

	def shift_and_predict(self, x: torch.Tensor, depth: int) -> torch.Tensor:
		y_hat = super().shift_and_predict(x, depth)

		y = torch.zeros((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
		y[:, self.y_channel_map] = y_hat

		return y
