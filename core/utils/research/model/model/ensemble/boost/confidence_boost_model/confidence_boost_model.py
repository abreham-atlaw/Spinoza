import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import ReverseSoftmax
from core.utils.research.model.model.savable import SpinozaModule


class ConfidenceBoostModel(SpinozaModule):

	def __init__(
			self,
			pre_model: SpinozaModule,
			post_model: SpinozaModule,
			confidence_model: SpinozaModule,
			confidence_range: typing.Tuple[float, float],
			softmax: bool = False
	):
		self.args = {
			"pre_model": pre_model,
			"post_model": post_model,
			"confidence_model": confidence_model,
			"confidence_range": confidence_range,
			"softmax": softmax
		}
		super().__init__(input_size=pre_model.input_size, auto_build=False)
		self.pre_model = pre_model
		self.post_model = post_model
		self.confidence_model = confidence_model
		self.confidence_range = tuple(confidence_range)
		self.softmax = nn.Softmax(dim=-1)
		self.reverse_softmax = nn.Identity() if softmax else ReverseSoftmax()
		self.init()

	def _standardize_confidence(self, x: torch.Tensor) -> torch.Tensor:
		y = (x - self.confidence_range[0])/(self.confidence_range[1] - self.confidence_range[0])
		y[y < 0] = 0
		y[y > 1] = 1
		return y

	def _call_confidence_model(self, x: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
		x_confidence = torch.concatenate((x, y_hat), dim=-1)
		with torch.no_grad():
			y_hat_confidence = self.confidence_model(x_confidence)
		y_hat_confidence = self._standardize_confidence(y_hat_confidence)
		return y_hat_confidence

	def _apply_confidence(self, pre_y_hat: torch.Tensor, post_y_hat: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
		confidence = torch.reshape(confidence, (confidence.shape[0], 1, 1))

		pre_y_hat = self.softmax(pre_y_hat)
		post_y_hat = self.softmax(post_y_hat)

		return pre_y_hat*confidence + post_y_hat*(1-confidence)

	def call(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			pre_y_hat = self.pre_model(x)
			post_y_hat = self.post_model(x)
		confidence = self._call_confidence_model(x, pre_y_hat)
		y = self._apply_confidence(pre_y_hat, post_y_hat, confidence)
		return self.reverse_softmax(y)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
