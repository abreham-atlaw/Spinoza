import typing

import torch.nn as nn

from .service_provider import ServiceProvider
from core import Config


class ResearchProvider:

	@staticmethod
	def provide_profit_predictor() -> 'ProfitPredictor':
		from core.utils.research.data.collect.analysis import ProfitPredictor
		return ProfitPredictor()

	@staticmethod
	def provide_runner_stats_repository(branch=None) -> 'RunnerStatsRepository':
		from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
		from .service_provider import ServiceProvider

		if branch is None:
			branch = Config.RunnerStatsBranches.default

		return RunnerStatsRepository(
			client=ServiceProvider.provide_mongo_client(),
			profit_based_selection=False,
			branch=branch,
		)

	@staticmethod
	def provide_rs_blacklist_repository(rs_repo=None) -> 'RSBlacklistRepository':
		from core.utils.research.data.collect.blacklist_repository import RSBlacklistRepository
		if rs_repo is None:
			rs_repo = ResearchProvider.provide_runner_stats_repository()
		return RSBlacklistRepository.from_rs_repository(rs_repo)

	@staticmethod
	def provide_times_repository() -> 'TimesRepository':
		from core.utils.research.data.collect.sim_setup.times_repository import JsonTimesRepository
		return JsonTimesRepository(Config.OANDA_SIM_TIMES_PATH)

	@staticmethod
	def provide_rs_setup_manager() -> 'RSSetupManager':

		from core.utils.research.data.collect.sim_setup.rs_setup_manager import RSSetupManager
		return RSSetupManager(
			fs=ServiceProvider.provide_file_storage(Config.OANDA_SIM_MODEL_IN_PATH),
			rs_repo=ResearchProvider.provide_runner_stats_repository(),
			times_repo=ResearchProvider.provide_times_repository(),
			model_evaluator=ResearchProvider.provide_model_evaluator(),
			add_timestep_pls=Config.UPDATE_AGENT
		)

	@staticmethod
	def provide_rt_rs_setup_manager() -> 'RealTimeRSSetupManager':
		from core.utils.research.data.collect.sim_setup.rt_rs_setup_manager import RealTimeRSSetupManager
		return RealTimeRSSetupManager(
			fs=ServiceProvider.provide_file_storage(Config.OANDA_SIM_MODEL_IN_PATH),
			rs_repo=ResearchProvider.provide_runner_stats_repository(),
			model_evaluator=ResearchProvider.provide_model_evaluator(),
			accounts_repo=ServiceProvider.provide_resman(Config.ResourceCategories.OANDA_ACCOUNTS)
		)

	@staticmethod
	def provide_ims_repository(model_name: str, label: str) -> 'IMSRepository':
		from core.utils.research.training.data.repositories.ims_repository import IMSRepository
		from core.di import ServiceProvider

		return IMSRepository(
			model_name=model_name,
			label=label,
			sync_size=Config.IMS_SYNC_SIZE,
			fs=ServiceProvider.provide_file_storage(Config.IMS_REMOTE_PATH),
			tmp_path=Config.IMS_TEMP_PATH
		)

	@staticmethod
	def provide_default_trackers(model_name: str) -> typing.List['TorchTracker']:
		return []
		# from core.utils.research.training.trackers.stats_tracker import DynamicStatsTracker
		# from core.utils.research.training.trackers.stats_tracker import Keys, WeightsStatsTracker, GradientsStatsTracker
		#
		# return [
		# 	DynamicStatsTracker(
		# 		model_name=model_name,
		# 		label=key
		# 	)
		# 	for key in Keys.ALL
		# ] + [
		# 	WeightsStatsTracker(
		# 		model_name=model_name
		# 	),
		# 	GradientsStatsTracker(
		# 		model_name=model_name
		# 	)
		# ]
		#

	@staticmethod
	def provide_loss_function() -> nn.Module:
		from core.utils.research.losses import ProximalMaskedLoss
		return ProximalMaskedLoss(
			n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
			p=1,
			softmax=True,
			multi_channel=(Config.MARKET_STATE_USE_MULTI_CHANNELS and (len(Config.MARKET_STATE_SIMULATED_CHANNELS) > 1))
		)

	@staticmethod
	def provide_model_evaluator(data_path: str = None) -> 'ModelEvaluator':
		if data_path is None:
			data_path = Config.UPDATE_SAVE_PATH

		from core.utils.research.utils.model_evaluator import ModelEvaluator
		return ModelEvaluator(
			data_path=data_path,
			cls_loss_fn=ResearchProvider.provide_loss_function(),
			batch_size=32
		)
