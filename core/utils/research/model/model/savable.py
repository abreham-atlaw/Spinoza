import importlib
import typing
from abc import ABC, abstractmethod

import torch
from torch import nn

from core.utils.research.utils.module_cache import ModuleCache
from lib.utils.cache import Cache
from lib.utils.logger import Logger


class SpinozaModule(nn.Module, ABC):

	def __init__(
			self,
			*args,
			input_size=None,
			output_size=None,
			auto_build=True,
			caching: bool = False,
			**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.built = False

		if isinstance(input_size, int):
			input_size = (None, input_size)
		if isinstance(input_size, list):
			input_size = tuple(input_size)

		self.output_size = output_size
		self.input_size = input_size
		if input_size is not None and auto_build:
			self.init()
		self.__state_dict_params = None
		self.__caching = caching
		self.__module_cache = ModuleCache(cache_size=5)

	def init(self):
		init_data = torch.rand((1,) + self.input_size[1:])
		out = self(init_data)
		self.output_size = (None,) + tuple(out.size()[1:])

	def set_caching(self, caching: bool, size=None):
		self.__caching = caching
		if size is not None:
			self.__module_cache = ModuleCache(cache_size=size)

	def _build(self, input_size: torch.Size):
		Logger.info(f"[{self.__class__.__name__}] Building...")
		self.input_size = (None, ) + tuple(input_size[1:])
		self.build(input_size)
		self.built = True
		if self.__state_dict_params is not None:
			Logger.info(f"[{self.__class__.__name__}] Loading state dict...")
			self.load_state_dict(*self.__state_dict_params[0], ** self.__state_dict_params[1])
			self.__state_dict_params = None

	@staticmethod
	def _get_input_size(*args, **kwargs) -> torch.Size:
		return args[0].size()

	def __cache(self, x, y):
		if self.__caching and not self.training:
			self.__module_cache.store(x, y)

	def __retrieve_cached(self, x):
		if self.__caching and not self.training:
			return self.__module_cache.retrieve(x)

	def build(self, input_size: torch.Size):
		pass

	@abstractmethod
	def call(self, *args, **kwargs) -> torch.Tensor:
		pass

	def forward(self, *args, **kwargs) -> torch.Tensor:
		if not self.built:
			self._build(self._get_input_size(*args, **kwargs))

		cached = self.__retrieve_cached(args)
		if cached is not None:
			# print(f"[{self.__class__.__name__}] Cached")
			return cached

		y = self.call(*args, **kwargs)

		self.__cache(args, y)
		return y

	def load_state_dict_lazy(self, *args, **kwargs):
		self.__state_dict_params = args, kwargs
		self.built = False

	@abstractmethod
	def export_config(self) -> typing.Dict[str, typing.Any]:
		pass

	@classmethod
	def import_config(cls, config) -> typing.Dict[str, typing.Any]:
		if config.get('hidden_activation') and isinstance(config['hidden_activation'], str):
			hidden_activation_module = importlib.import_module('torch.nn')
			config['hidden_activation'] = getattr(hidden_activation_module, config['hidden_activation'])()
		if config.get('init_fn'):
			init_fn_module = importlib.import_module('torch.nn.init')
			config['init_fn'] = getattr(init_fn_module, config['init_fn'])
		return config


