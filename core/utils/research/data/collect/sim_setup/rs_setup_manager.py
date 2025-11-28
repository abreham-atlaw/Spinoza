import os
from datetime import datetime

from torch import nn

from core import Config
from core.utils.misc.sim_trading.setup import SetupManager
from core.utils.research.data.collect.runner_stats import RunnerStats
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
from core.utils.research.data.collect.sim_setup.times_repository import TimesRepository
from core.utils.research.model.model.utils import TemperatureScalingModel, TransitionOnlyModel
from lib.utils.decorators import retry
from lib.utils.file_storage import FileStorage, FileNotFoundException
from lib.utils.logger import Logger
from lib.utils.torch_utils.model_handler import ModelHandler
from core.utils.research.utils.model_evaluator import ModelEvaluator


class RSSetupManager:

	def __init__(
			self,
			times_repo: TimesRepository,
			rs_repo: RunnerStatsRepository,
			fs: FileStorage,
			model_evaluator: ModelEvaluator
	):
		self.__times_repo = times_repo
		self.__rs_repo = rs_repo
		self.__fs = fs
		self._setup_manager = self._init_setup_manager()
		self.__model_evaluator = model_evaluator

	def _init_setup_manager(self) -> SetupManager:
		return SetupManager()

	def __serialize_time(self, time: datetime):
		return time.strftime("%Y-%m-%d %H:%M:%S+00:00")

	def _allocate_time(self, stat: RunnerStats) -> datetime:
		return min(
			self.__times_repo.retrieve_all(),
			key=lambda t: stat.simulated_timestamps.count(self.__serialize_time(t))
		)

	def _allocate_extra(self, stat: RunnerStats):
		Logger.info(f"Allocating Time...")
		start_time = self._allocate_time(stat)
		Logger.info(f"Allocated Start Time: {start_time}")
		stat.simulated_timestamps.append(self.__serialize_time(start_time))

		Logger.info(f"Setting up Simulation Trading...")
		self._setup_manager.setup(start_time)

	@retry(exception_cls=(FileNotFoundException,), patience=10)
	def setup(self):
		Logger.info(f"Allocating Runner Stat...")
		stat = self.__rs_repo.allocate_for_runlive()
		Logger.success(f"Allocated Runner Stat: {stat}")

		Logger.info(f"Downloading {stat.model_name}...")
		self.__fs.download(stat.model_name, os.path.abspath(stat.model_name))

		Config.CORE_MODEL_CONFIG.path = os.path.abspath(stat.model_name)
		Config.AGENT_MODEL_TEMPERATURE = stat.temperature
		Logger.success(f"Downloaded {stat.model_name} to {Config.CORE_MODEL_CONFIG.path}")

		self._allocate_extra(stat)

		Logger.success(f"Setup Complete!")

		return stat

	@staticmethod
	def __load_model(path: str, temperature: float) -> nn.Module:
		model = TemperatureScalingModel(
			ModelHandler.load(path),
			temperature=temperature
		)
		tom_model = TransitionOnlyModel(
			model=model,
			extra_len=Config.AGENT_MODEL_EXTRA_LEN
		)
		tom_model.input_size = model.input_size
		tom_model.export_config = model.export_config
		return tom_model

	def __evaluate_model_loss(self, model_path: str, temperature: float) -> float:
		Logger.info(f"Evaluating Model Loss...")
		model = self.__load_model(path=model_path, temperature=temperature)
		losses = self.__model_evaluator.evaluate(model)
		return losses[0]

	def _finish_extra(self, stat: RunnerStats):
		pass

	def finish(
			self,
			stat: RunnerStats,
			pl: float,
			model_path: str = None
	):
		if model_path is None:
			model_path = Config.CORE_MODEL_CONFIG.path

		Logger.info(f"Finishing Session...")

		Logger.info(f"Session PL: {pl}")
		stat.add_profit(pl)
		stat.add_duration((datetime.now() - stat.session_timestamps[-1]).total_seconds())

		session_model_loss = self.__evaluate_model_loss(
			model_path=model_path,
			temperature=stat.temperature
		)
		Logger.info(f"Session Model Loss: {session_model_loss}")
		stat.add_session_model_loss(session_model_loss)

		Logger.info(f"Storing Session...")
		self.__rs_repo.store(stat)

		self._finish_extra(stat)

		Logger.success(f"Finished Session!")
