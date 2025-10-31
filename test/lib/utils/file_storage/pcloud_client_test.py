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

