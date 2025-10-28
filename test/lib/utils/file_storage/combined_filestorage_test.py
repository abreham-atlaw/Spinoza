import unittest

from core import Config
from core.di import ServiceProvider


class FileStorageTest(unittest.TestCase):

	def setUp(self):
		self.fs = ServiceProvider.provide_file_storage(path="/")

	def test_listdir(self):
		files = self.fs.listdir("/")
		self.assertTrue(len(files) > 0)

	def test_delete(self):
		FILES = [
			"/0.txt",
			"/1.txt",
			"/2.txt"
		]
		files = self.fs.listdir("/")

		for file in FILES:
			self.assertTrue(file in files)
		for file in FILES:
			self.fs.delete(file)
		files = self.fs.listdir("/")
		for file in FILES:
			self.assertTrue(file not in files)

	def test_create_folder(self):
		self.fs.mkdir("/test")
		files = self.fs.listdir("/")
		self.assertTrue("/test" in files)

	def test_get_metadata(self):
		metadata = self.fs.get_metadata("/test/0.txt")
		print(metadata)
		self.assertIsNotNone(metadata.size)

	def test_get_url(self):
		file = "/Apps/RTrader/abrehamalemu-spinoza-lass-training-cnn-2-it-11-tot.zip"
		print(self.fs.get_url(file))
