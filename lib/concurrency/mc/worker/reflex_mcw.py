from abc import ABC

from lib.rl.agent import Node
from lib.rl.agent.mca import ReflexMonteCarloAgent
from .mcw import MonteCarloWorkerAgent


class ReflexMonteCarloWorker():

	def _prepare_reflex_action(self):
		return

	def _finalize_step(self, root: Node):
		MonteCarloWorkerAgent._finalize_step(self, root)
