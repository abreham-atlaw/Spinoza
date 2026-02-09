import os.path
import unittest

from core import Config
from lib.utils.file_storage import PCloudClient
from lib.utils.logger import Logger


class PCloudClientTest(unittest.TestCase):

	def test_get_quotas(self):

		clients = [
			PCloudClient(token, "/")
			for token in Config.PCLOUD_TOKENS
		]

		for i, client in enumerate(clients):
			Logger.info(f"Client {i}: {client.get_quota_usage()}")

	def test_get_find_file(self):
		clients = [
			PCloudClient(token, "/")
			for token in Config.PCLOUD_TOKENS
		]

		filepath = "/Apps/RTrader/abrehamalemu-spinoza-lass-training-cnn-17-it-11-tot.1.zip"

		for i, client in enumerate(clients):
			files = client.listdir(os.path.dirname(filepath))
			if filepath in files:
				Logger.info(f"Found File in Client: {i}")
				break
