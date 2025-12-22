import typing
import torch

from core.utils.research.model.model.savable import SpinozaModule


class MCInputPadding(SpinozaModule):

	def __init__(self, *args, channels: typing.Tuple[int, ...] = None, **kwargs):
		self.args = {
			"channels": channels
		}
		super().__init__(*args, **kwargs)
		self.channels = channels

	def get_channels(self, x: torch.Tensor):
		if self.channels is None:
			return torch.arange(x.shape[1], device=x.device)
		return torch.as_tensor(self.channels, device=x.device)

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x = x.clone()

		channels = self.get_channels(x)
		y = x[:, channels, :]

		non_zero = y != 0

		t = torch.arange(y.size(-1), device=y.device)
		t = t.view(1, 1, -1)

		idx = t * non_zero

		last_valid_idx = idx.cummax(dim=-1).values

		y_filled = torch.gather(y, dim=-1, index=last_valid_idx)

		x[:, channels, :] = y_filled

		return x

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return dict(self.args)
