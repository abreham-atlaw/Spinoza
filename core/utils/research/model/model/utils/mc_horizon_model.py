import torch

from lib.utils.logger import Logger
from .horizon_model import HorizonModel


class MCHorizonModel(HorizonModel):

	def __init__(self, *args, close_channel: int = 0, **kwargs):
		super().__init__(*args, **kwargs)
		Logger.info(f"Initializing MCHorizonModel...")
		self.args.update({
			"close_channel": close_channel
		})
		self.close_channel = close_channel

	def _retrieve_recent_close(self, x: torch.Tensor):
		return  x[..., self.close_channel,  -(self.X_extra_len + 1)]

	def shift_and_predict(self, x: torch.Tensor, depth: int) -> torch.Tensor:
		y = super().shift_and_predict(x, depth)
		y = torch.concatenate(
			(
				torch.unsqueeze(y, dim=1),
				torch.zeros((x.shape[0], x.shape[1]-1), device=x.device)
			),
			dim=1
		)
		return y
