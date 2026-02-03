import unittest

from lib.concurrency.swarm.swarm_socket import SwarmSocket


class SwarmSocketTest(unittest.TestCase):

	def setUp(self):
		self.sio = SwarmSocket()

	def test_connect(self):
		self.sio.connect(f"http://127.0.0.1:8888")
		self.sio.emit("create-session")
		self.sio.disconnect()
		self.sio.reset()
		self.sio.connect(f"http://127.0.0.1:8888")
		self.sio.emit("create-session")

