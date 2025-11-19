import os

from core.di import ServiceProvider
from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.file_storage import FileStorage
from lib.utils.torch_utils.model_handler import ModelHandler


class ModelUtils:

	@staticmethod
	def load_from_fs(path: str, file_storage: FileStorage = None) -> SpinozaModule:
		local_path = os.path.basename(path)
		if not os.path.exists(local_path):
			if file_storage is None:
				file_storage = ServiceProvider.provide_file_storage()
			file_storage.download(path, local_path)
		return ModelHandler.load(local_path)
