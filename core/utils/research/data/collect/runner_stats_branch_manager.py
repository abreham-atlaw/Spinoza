import typing
from copy import deepcopy

from core.Config import RunnerStatsBranches
from core.di import ResearchProvider
from core.utils.research.data.collect.runner_stats import RunnerStats
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
from lib.utils.logger import Logger


class RunnerStatsBranchManager:

	def __init__(
			self,
			all_branches: typing.List[str] = None,
			sync_model_losses: bool = False,
			stat_filter: typing.Callable = None
	):
		if all_branches is None:
			all_branches = RunnerStatsBranches.all
		if stat_filter is None:
			stat_filter = lambda stat: True
		self.__all_branches = all_branches
		self.__repositories = {}
		self.__is_sync_model_losses = sync_model_losses
		self.__stat_filter = stat_filter

	def __get_repository(self, branch: str) -> RunnerStatsRepository:
		cached = self.__repositories.get(branch)
		if cached is None:
			self.__repositories[branch] = ResearchProvider.provide_runner_stats_repository(branch)
			return self.__get_repository(branch)
		return cached

	def __allow_model_loss_sync(self, source_stat: RunnerStats, target_stat: RunnerStats):
		source_losses, target_losses = [
			list(filter(lambda x: x > 0.0, stat.model_losses))
			for stat in [source_stat, target_stat]
		]
		return source_losses > target_losses

	def __sync_model_losses(self, source_stat: RunnerStats, target_stat: RunnerStats, repo: RunnerStatsRepository):

		if not self.__allow_model_loss_sync(source_stat, target_stat):
			return

		Logger.info(f"Syncing Model Losses from '{source_stat.id}' -> '{target_stat.id}'")
		target_stat.model_losses = source_stat.model_losses
		repo.store(target_stat)

	def __filter_stats(self, stats: typing.List[RunnerStats]) -> typing.List[RunnerStats]:
		Logger.info(f"Filtering {len(stats)} stats...")
		return list(filter(
			self.__stat_filter,
			stats
		))

	def __sync_branch(
			self,
			branch: str,
			synced_stats: typing.List[RunnerStats],
			branches: typing.List[str]
	) -> typing.List[RunnerStats]:

		synced_ids = [stat.id for stat in synced_stats]

		stats = self.__get_repository(branch=branch).retrieve_all()
		stats = self.__filter_stats(stats)
		Logger.info(f"Syncing {len(stats)} stats...")

		for target_branch in branches:
			if target_branch == branch:
				continue

			Logger.info(f"Syncing '{branch}' -> '{target_branch}'")

			target_repository = self.__get_repository(branch=target_branch)
			target_stats = target_repository.retrieve_all()
			target_ids = [stat.id for stat in target_stats]

			for i, stat in enumerate(stats):
				Logger.info(f"Processing {(i+1)*100/len(stats): .2f}%...")
				if stat.id in synced_ids:
					continue

				if stat.id in target_ids:
					if self.__is_sync_model_losses:
						self.__sync_model_losses(stat, target_stats[target_ids.index(stat.id)], target_repository)
					continue

				new_stat = deepcopy(stat)
				(
					new_stat.session_timestamps,
					new_stat.profits,
					new_stat.real_profits,
					new_stat.duration,
					new_stat.branch,
					new_stat.simulated_timestamps,
					new_stat.session_model_losses
				) = (
					[],
					[],
					[],
					0,
					target_branch,
					[],
					[]
				)
				self.__get_repository(target_branch).store(new_stat)
				Logger.info(f"Added {new_stat.id} to {target_branch}")

		return stats

	def sync_branches(self, branches: typing.List[str] = None):

		if branches is None:
			branches = self.__all_branches

		synced_stats = []

		for source_stat in branches:

			new_syncs = self.__sync_branch(source_stat, synced_stats, branches)
			synced_stats.extend(new_syncs)
