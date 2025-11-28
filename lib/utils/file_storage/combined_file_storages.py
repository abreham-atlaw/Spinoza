import os.path
import random
import typing

import numpy as np

from .file_storage import FileStorage, MetaData
from .pcloud_client import PCloudClient
from .exceptions import FileNotFoundException
from lib.utils.logger import Logger


class CombinedFileStorage(FileStorage):

	def __init__(self, children: typing.List[FileStorage]):
		self.__children = children

	def _get_storage(self, path) -> FileStorage:
		for i, child in enumerate(self.__children):
			try:
				child.get_url(path)
				Logger.info(f"Using Storage {i} for {path}")
				return child
			except FileNotFoundException:
				pass
		raise FileNotFoundException(path)

	def _random_storage_choice(self) -> FileStorage:
		weights = 1/np.array([child.get_quota_usage() for child in self.__children])
		weights = weights/np.sum(weights)
		return random.choices(self.__children, weights=list(weights))[0]

	def _choose_storage(self, file_path: str, upload_path: typing.Union[str, None] = None):
		try:
			path = upload_path
			if upload_path is None:
				path = os.path.basename(file_path)
			return self._get_storage(path)
		except FileNotFoundException:
			fs = self._random_storage_choice()
			Logger.info(f"Selected Storage {self.__children.index(fs)} for {path}...")
			return fs

	def get_url(self, path) -> str:
		storage = self._get_storage(path)
		return storage.get_url(path)

	def upload_file(self, file_path: str, upload_path: typing.Union[str, None] = None):
		storage = self._choose_storage(file_path, upload_path)
		storage.upload_file(file_path, upload_path)

	def listdir(self, path: str) -> typing.List[str]:
		files = []
		for child in self.__children:
			try:
				child_files = child.listdir(path)
				files.extend(child_files)
			except FileNotFoundException:
				pass
		return sorted(list(set(files)))

	def delete(self, path: str):
		Logger.info(f"Deleting {path}...")
		for i, child in enumerate(self.__children):
			try:
				child.delete(path)
				Logger.info(f"Deleted from Storage {i} for {path}")
			except FileNotFoundException:
				pass

	def mkdir(self, path: str):
		for child in self.__children:
			child.mkdir(path)

	def get_metadata_raw(self, path: str) -> typing.Dict[str, typing.Any]:
		storage = self._get_storage(path)
		return storage.get_metadata_raw(path)

	def get_metadata(self, path: str) -> MetaData:
		storage = self._get_storage(path)
		return storage.get_metadata(path)


class PCloudCombinedFileStorage(CombinedFileStorage):

	def __init__(self, tokens: typing.List[str], base_path: str):
		super().__init__(
			[
				PCloudClient(token, base_path)
				for token in tokens
			]
		)
