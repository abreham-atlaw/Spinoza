import typing
from multiprocessing import Process

from core.agent.concurrency.swarm.worker import SwarmWorkerTrader
from lib.concurrency.swarm.worker import SwarmWorker
from lib.rl.environment import Environment
from lib.utils.logger import Logger


class SwarmWorkerProcess(Process):

	def __init__(self):
		super().__init__()
		self.__worker = None
		self.__environment = None

	def set_environment(self, environment: Environment):
		self.__environment = environment

	def run(self):
		self.__worker = SwarmWorkerTrader()
		self.__worker.set_environment(self.__environment)
		self.__worker.loop()


class SwarmWorkerPool:

	def __init__(self, workers: int = 5):
		self.__workers = self.__create_workers(workers)

	def __create_workers(self, n: int) -> typing.List[SwarmWorkerProcess]:
		return [
			SwarmWorkerProcess()
			for _ in range(n)
		]

	def set_environment(self, environment: Environment):
		for worker in self.__workers:
			worker.set_environment(environment)

	def start(self):
		for i, worker in enumerate(self.__workers):
			Logger.info(f"Starting worker {i}...")
			worker.start()

		for worker in self.__workers:
			worker.join()
