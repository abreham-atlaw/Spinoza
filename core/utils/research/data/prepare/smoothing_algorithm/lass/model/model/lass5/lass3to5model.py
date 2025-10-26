import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class Lass3To5Model(SpinozaModule):

	def __init__(self, model: SpinozaModule):
		self.args = {
			'model': model
		}
		super().__init__(input_size=(None, model.input_size[-1]), auto_build=False)
		self.model = model
		self.init()

	@staticmethod
	def _prepare_input(x: torch.Tensor, y_hat: typing.List[torch.Tensor], i: int) -> torch.Tensor:
		inputs = torch.zeros((x.size(0), 2, x.size(1)), device=x.device)
		inputs[:, 0] = x
		if i > 0:
			inputs[:, 1, -i:] = y_hat
		return inputs

	def call(self, x: torch.Tensor) -> torch.Tensor:
		y_hat = []
		for i in range(x.size(1)):
			inputs = self._prepare_input(
				x,
				torch.stack(y_hat, dim=1) if y_hat else None,
				i
			)
			prediction = self.model(inputs)
			y_hat.append(torch.reshape(prediction, (x.size(0),)))

		y_hat = torch.stack(y_hat, dim=1)
		return y_hat

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
