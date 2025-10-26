import hashlib
import io
import shutil
import typing
from uuid import uuid4

import torch
import json
import zipfile
import os
import importlib

from torch import nn

from core.utils.research.model.model.savable import SpinozaModule
from lib.network.rest_interface.pickle_serializer import PickleSerializer
from lib.utils.logger import Logger
from lib.utils.torch_utils.module_serializer import TorchModuleSerializer


class ModelHandler:

	__MODEL_PREFIX = "__model__"
	__MODULE_PREFIX = "__module__"
	__OBJECT_PREFIX = "__object__"

	__module_serializer = TorchModuleSerializer()
	__pickle_serializer = PickleSerializer()

	@staticmethod
	def get_model_device(model):
		try:
			return next(model.parameters()).device
		except StopIteration:
			try:
				return next(model.buffers()).device
			except StopIteration:
				raise ValueError("The model has no parameters or buffers to infer the device.")

	@staticmethod
	def __export_torch_module(item: nn.Module, key: str, idx: int) -> str:
		return ModelHandler.__module_serializer.serialize(item)

	@staticmethod
	def __export_spinoza_module(item: SpinozaModule, key: str, idx: int = None) -> str:
		filename = f"{key}_{idx}.zip" if idx is not None else f"{key}.zip"
		ModelHandler.save(item, filename, save_state=False)
		return filename

	@staticmethod
	def __export_object(item: object, key: str, idx: int = None) -> str:
		return ModelHandler.__pickle_serializer.serialize(item)

	@staticmethod
	def __requires_serialization(item: object) -> bool:
		try:
			json.dumps(item)
			return False
		except (TypeError, OverflowError):
			return True

	@staticmethod
	def __export_item(
			item: typing.Union[nn.Module, SpinozaModule, typing.Any, typing.List],
			key: str,
			idx: int = None
	) -> typing.Tuple[str, typing.Any]:

		if isinstance(item, typing.List) and len(item) > 0:

			return (
				f"{ModelHandler.__MODEL_PREFIX}{key}" if isinstance(item[0], SpinozaModule)
				else f"{ModelHandler.__MODULE_PREFIX}{key}" if isinstance(item[0], nn.Module)
				else f"{ModelHandler.__OBJECT_PREFIX}{key}" if ModelHandler.__requires_serialization(item[0])
				else key,

				[
					ModelHandler.__export_item(i, key, idx)[1]
					for idx, i in enumerate(item)
				]
			)

		if isinstance(item, SpinozaModule):
			return f"{ModelHandler.__MODEL_PREFIX}{key}", ModelHandler.__export_spinoza_module(item, key, idx)
		if isinstance(item, nn.Module):
			return f"{ModelHandler.__MODULE_PREFIX}{key}", ModelHandler.__export_torch_module(item, key, idx)
		if ModelHandler.__requires_serialization(item):
			return f"{ModelHandler.__OBJECT_PREFIX}{key}", ModelHandler.__export_object(item, key, idx)
		else:
			return key, item

	@staticmethod
	def __import_torch_module(serialized: str) -> nn.Module:
		return ModelHandler.__module_serializer.deserialize(serialized)

	@staticmethod
	def __import_spinoza_module(filename: str, dirname: str) -> SpinozaModule:
		return ModelHandler.load(os.path.join(dirname, filename), load_state=False)

	@staticmethod
	def __import_object(serialized: str) -> object:
		return ModelHandler.__pickle_serializer.deserialize(serialized)

	@staticmethod
	def __import_item(key, value, dirname: str) -> typing.Tuple[str, typing.Any]:

		if isinstance(value, list):
			return (
				key[len(ModelHandler.__MODEL_PREFIX):] if key.startswith(ModelHandler.__MODEL_PREFIX) else
				key[len(ModelHandler.__MODULE_PREFIX):] if key.startswith(ModelHandler.__MODULE_PREFIX) else
				key[len(ModelHandler.__OBJECT_PREFIX):] if key.startswith(ModelHandler.__OBJECT_PREFIX) else
				key,
				[
					ModelHandler.__import_item(key, i, dirname)[1]
					for i in value
				]
			)

		if key.startswith(ModelHandler.__MODEL_PREFIX):
			return key[len(ModelHandler.__MODEL_PREFIX):], ModelHandler.__import_spinoza_module(value, dirname)
		if key.startswith(ModelHandler.__MODULE_PREFIX):
			return key[len(ModelHandler.__MODULE_PREFIX):], ModelHandler.__import_torch_module(value)
		if key.startswith(ModelHandler.__OBJECT_PREFIX):
			return key[len(ModelHandler.__OBJECT_PREFIX):], ModelHandler.__import_object(value)
		else:
			return key, value

	@staticmethod
	def save(model, path, to_cpu=True, save_state=True):
		Logger.info(f"Saving {model.__class__.__name__} to {path}...")

		path = os.path.abspath(path)
		cwd = os.path.join(os.path.dirname(path), os.path.basename(path).replace(".", "_"))
		os.makedirs(cwd, exist_ok=True)
		os.chdir(cwd)

		original_device = None
		if to_cpu:
			try:
				original_device = ModelHandler.get_model_device(model)
			except ValueError:
				pass
			model = model.to(torch.device('cpu'))

		model_config = model.export_config()

		model_config['class_name'] = model.__class__.__name__
		model_config['module_name'] = model.__class__.__module__

		model_config_copy = {}
		for key, value in model_config.items():

			export_key, export_value = ModelHandler.__export_item(value, key)
			model_config_copy[export_key] = export_value

		model_config = model_config_copy

		with open('model_config.json', 'w') as f:
			json.dump(model_config, f)

		if save_state:
			torch.save(model.state_dict(), 'model_state.pth')

		with zipfile.ZipFile(path, 'w') as zipf:
			zipf.write('model_config.json')
			if save_state:
				zipf.write('model_state.pth')
			for key, value in model_config.items():
				if key.startswith(ModelHandler.__MODEL_PREFIX):
					if isinstance(value, (list, tuple)):
						for i in range(len(value)):
							zipf.write(value[i])
					else:
						zipf.write(value)

		os.remove('model_config.json')
		if save_state:
			os.remove('model_state.pth')
		for key, value in model_config.items():
			if key.startswith(ModelHandler.__MODEL_PREFIX):
				if isinstance(value, (list, tuple)):
					for i in range(len(value)):
						os.remove(value[i])
				else:
					os.remove(value)

		if to_cpu and original_device is not None:
			model.to(original_device)

		os.chdir("..")
		os.rmdir(cwd)

	@staticmethod
	def load(path, dtype=None, load_state=True):
		dirname = f"{os.path.basename(path).replace('.', '_')} - {uuid4()}"

		try:
			os.makedirs(dirname)
		except FileExistsError:
			pass

		with zipfile.ZipFile(path, 'r') as zipf:
			zipf.extractall(dirname, )

		with open(os.path.join(dirname, 'model_config.json'), 'r') as f:
			model_config = json.load(f)

		module = importlib.import_module(model_config['module_name'])
		ModelClass = getattr(module, model_config['class_name'])

		model_config.pop('class_name')
		model_config.pop('module_name')

		model_config_copy = {}
		for key, value in model_config.items():

			imported_key, imported_value = ModelHandler.__import_item(key, value, dirname)
			model_config_copy[imported_key] = imported_value

		model_config = model_config_copy
		model_config = ModelClass.import_config(model_config)

		model: SpinozaModule = ModelClass(**model_config)

		if load_state:
			model.load_state_dict_lazy(torch.load(os.path.join(dirname, 'model_state.pth'), map_location=torch.device('cpu')))

		shutil.rmtree(dirname, ignore_errors=True)

		if dtype is not None:
			model = model.to(dtype)
		return model

	@staticmethod
	def generate_signature(model: SpinozaModule) -> str:
		config = model.export_config()
		return hashlib.sha256(
			f"{model.__class__.__name__}-{model.__class__.__module__}-{str(config)}".encode("utf-8")
		).hexdigest()
