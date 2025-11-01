import typing
from typing import *
from abc import ABC, abstractmethod

import dropbox

import random
import os
from threading import Thread

from lib.network.rest_interface.requests import Request
from lib.network.rest_interface.NetworkApiClient import NetworkApiClient
from .exceptions import FileNotFoundException, FileSystemException
from lib.utils.logger import Logger
from .file_storage import FileStorage, MetaData


class PCloudClient(FileStorage):

	class PCloudNetworkClient(NetworkApiClient):

		def __init__(self, token: str, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.__token = token

		def execute(self, request: Request, headers: Optional[Dict] = None):
			request.get_get_params()["auth"] = self.__token
			return super().execute(request, headers)

	class UploadFileRequest(Request):

		def __init__(self, filepath: str, upload_path: Optional[str] = None):
			get_params = {}
			if upload_path is not None:
				get_params["path"] = upload_path
			super().__init__(
				"uploadfile",
				method=Request.Method.POST,
				get_params=get_params,
				files={
					"file": filepath
				},
				headers={
					"Content-Type": None
				}
			)

	class CreateUrlRequest(Request):

		def __init__(self, path):
			super().__init__(
				"/getfilepublink",
				method=Request.Method.GET,
				get_params={
					"path": path
				},
				output_class=str
			)

		def _filter_response(self, response: Dict) -> str:
			return response.get("code")

	class GetUrlRequest(Request):

		def __init__(self, code: str, path: str):
			super().__init__(
				"/getpublinkdownload",
				method=Request.Method.GET,
				get_params={
					"code": code
				},
				output_class=str
			)
			self.__path = path

		def _filter_response(self, response):
			if isinstance(response, dict) and response.get("result") == 7001:
				raise FileNotFoundException(self.__path)
			return f"{response.get('hosts')[0]}{response.get('path')}".replace('\/', '/')

	class ListDirRequest(Request):

		def __init__(self, path: str):
			super().__init__(
				"/listfolder",
				method=Request.Method.GET,
				get_params={
					"path": path
				},
				output_class=typing.List[str]
			)

		def _filter_response(self, response):
			return response["metadata"]["contents"]

		def deserialize_object(self, response) -> object:
			return [
				content["path"]
				for content in self._filter_response(response)
			]

	class DeleteFileRequest(Request):

		def __init__(self, path: str):
			super().__init__(
				"/deletefile",
				method=Request.Method.POST,
				get_params={
					"path": path
				}
			)
			self.__path = path

		def _filter_response(self, response):
			if response["result"] == 0:
				return response

			if response["result"] == 2009:
				raise FileNotFoundException(self.__path)

			raise FileSystemException(f"Failed to delete file of path={self.__path}. {response}")

	class CreateFolderRequest(Request):

		def __init__(self, path: str):
			super().__init__(
				"/createfolderifnotexists",
				method=Request.Method.POST,
				get_params={
					"path": path
				}
			)
			self.__path = path

		def _filter_response(self, response):
			if response["result"] == 0:
				return response

			raise FileSystemException(f"Failed to create folder of path={self.__path}. {response}")

	class GetMetadataRequest(Request):

		def __init__(self, path: str):
			super().__init__(
				"/stat",
				method=Request.Method.GET,
				get_params={
					"path": path
				}
			)
			self.__path = path

		def _filter_response(self, response):
			if response["result"] == 0:
				return response["metadata"]

			raise FileSystemException(f"Failed to get metadata of path={self.__path}. {response}")

	class GetQuotaUsage(Request):

		def __init__(self):
			super().__init__(
				"/userinfo",
				method=Request.Method.GET,
				output_class=float
			)

		def _filter_response(self, response):
			return float(response["usedquota"])/float(response["quota"])

	def __init__(self, token, folder, pcloud_base_url="https://api.pcloud.com/"):
		self.__base_path = folder
		self.__client = PCloudClient.PCloudNetworkClient(token=token, url=pcloud_base_url)

	def __get_complete_path(self, path: str) -> str:
		return os.path.normpath(os.path.join(self.__base_path, path))

	def get_url(self, path) -> str:
		code = self.__client.execute(
			PCloudClient.CreateUrlRequest(
				self.__get_complete_path(path)
			)
		)
		return self.__client.execute(
			PCloudClient.GetUrlRequest(
				code,
				path
			)
		)

	def upload_file(self, file_path: str, upload_path: Union[str, None] = None):
		if upload_path is None:
			upload_path = ""
		Logger.info(f"Uploading {file_path} => {self.__get_complete_path(upload_path)}")
		self.__client.execute(
			PCloudClient.UploadFileRequest(
				file_path,
				self.__get_complete_path(upload_path)
			)
		)

	def listdir(self, path: str) -> typing.List[str]:
		try:
			return self.__client.execute(
				PCloudClient.ListDirRequest(
					self.__get_complete_path(path)
				)
			)
		except KeyError as ex:
			raise FileNotFoundException(path)

	def delete(self, path: str):
		self.__client.execute(
			PCloudClient.DeleteFileRequest(
				self.__get_complete_path(path)
			)
		)

	def mkdir(self, path: str):
		self.__client.execute(
			PCloudClient.CreateFolderRequest(
				self.__get_complete_path(path)
			)
		)

	def get_metadata_raw(self, path: str) -> typing.Dict[str, typing.Any]:
		return self.__client.execute(
			PCloudClient.GetMetadataRequest(
				self.__get_complete_path(path)
			)
		)

	def get_metadata(self, path: str) -> MetaData:
		data = self.get_metadata_raw(path)
		return MetaData(
			size=data["size"]
		)

	def get_quota_usage(self) -> float:
		return self.__client.execute(
			PCloudClient.GetQuotaUsage()
		)
