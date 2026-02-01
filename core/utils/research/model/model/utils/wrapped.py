import torch
from torch import nn

from core.utils.research.model.layers import MovingAverage


class WrappedModel(nn.Module):

	def __init__(
			self,
			model: nn.Module,
			seq_len: int = None,
			window_size: int = None,
			use_ma: bool = False,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		if use_ma and (window_size is None or seq_len is None):
			raise Exception("window_size and seq_len must be provided if use_ma is True")
		self.model = model
		self.seq_len = seq_len
		if use_ma:
			self.ma = MovingAverage(window_size)
		self.softmax = nn.Softmax(dim=-1)
		self.use_ma = use_ma

	def forward(self, inputs: torch.Tensor):
		if self.use_ma:
			inputs = torch.concat([self.ma(inputs[:, :self.seq_len]), inputs[:, self.seq_len:]], dim=1)
		outputs = self.model(inputs)
		return torch.concat([self.softmax(outputs[..., :-1]), outputs[..., -1:]], dim=-1)
