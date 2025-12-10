import typing
from dataclasses import dataclass
from typing import *
from abc import ABC, abstractmethod

import random
import os
from threading import Thread

from .exceptions import FileNotFoundException, FileSystemException


@dataclass
class MetaData:
	size: int


class FileStorage(ABC):

	@abstractmethod
	def get_url(self, path) -> str:
		pass

	@abstractmethod
	def upload_file(self, file_path: str, upload_path: Union[str, None] = None):
		pass

	@abstractmethod
	def listdir(self, path: str) -> typing.List[str]:
		pass

	@abstractmethod
	def delete(self, path: str):
		pass

	@abstractmethod
	def mkdir(self, path: str):
		pass

	@abstractmethod
	def get_metadata_raw(self, path: str) -> typing.Dict[str, typing.Any]:
		pass

	@abstractmethod
	def get_metadata(self, path: str) -> MetaData:
		pass

	def get_quota_usage(self) -> float:
		return 0.0

	def exists(self, path: str) -> bool:
		try:
			files = self.listdir(os.path.dirname(path))
			return os.path.basename(path) in files
		except FileNotFoundException:
			return False

	def download(self, path, download_path: Union[str, None] = None):
		url = self.get_url(path)
		command = f"wget --no-verbose \"{url}\""
		if download_path is not None:
			command = f"{command} -O {download_path}"
		os.system(command)




