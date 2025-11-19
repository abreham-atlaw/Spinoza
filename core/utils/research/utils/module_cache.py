import typing

import torch

from lib.utils.cache import Cache


class ModuleCache(Cache):

	@staticmethod
	def _hash(value: typing.Tuple[torch.Tensor, ...]) -> int:
		return hash("\n".join([
			f"{i}-{v.shape}-{v.dtype}-{torch.sum(v)}-{torch.mean(v)}-{torch.std(v)}"
			for i, v in enumerate(value)
		]))
