from abc import ABC

from lib.concurrency.swarm.worker.swarm_worker import SwarmWorker
from lib.rl.agent import MonteCarloAgent
from lib.rl.agent.mca import ReflexMonteCarloAgent


class ReflexSwarmWorker(SwarmWorker, ReflexMonteCarloAgent, ABC):

	def _prepare_reflex_action(self):
		pass

	def _finalize_step(self, root: 'Node'):
		return MonteCarloAgent._finalize_step(self, root)
