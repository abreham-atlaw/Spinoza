import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class PadOverlayCombiner(SpinozaModule):

	@staticmethod
	def _get_input_size(*args, **kwargs) -> torch.Size:
		return args[0][0].size()

	def call(self, x: typing.List[torch.Tensor]) -> torch.Tensor:
		if isinstance(x, torch.Tensor) and len(x.shape) < 3:
			return x.unsqueeze(2)

		size = max([arr.shape[-1] for arr in x])

		y = torch.stack([
			torch.concatenate((
				torch.expand_copy(arr[:, :1], (-1, size-arr.shape[1])),
				arr
			), dim=1)
			for arr in x
		], dim=1)
		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return {}
