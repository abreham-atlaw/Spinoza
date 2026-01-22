import os.path
import random
import typing
from datetime import datetime

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from core import Config
from core.Config import MODEL_SAVE_EXTENSION, BASE_DIR
from core.di import ResearchProvider
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository, RunnerStats
from core.utils.research.losses import CrossEntropyLoss, ProximalMaskedLoss, MeanSquaredClassError, \
	PredictionConfidenceScore, \
	SpinozaLoss, ReverseMAWeightLoss, MultiLoss, ScoreLoss, SoftConfidenceScore, ProximalMaskedLoss2, \
	ProximalMaskedLoss3
from core.utils.research.model.model.utils import TemperatureScalingModel, HorizonModel, AggregateModel
from core.utils.research.utils.model_evaluator import ModelEvaluator
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.file_storage import FileStorage, FileNotFoundException
from lib.utils.fileio import load_json
from lib.utils.logger import Logger
from lib.utils.torch_utils.model_handler import ModelHandler
from .blacklist_repository import RSBlacklistRepository
from .runner_stats2 import RunnerStats2
from ..prepare.utils.data_prep_utils import DataPrepUtils


class RunnerStatsPopulater:

	def __init__(
			self,
			repository: RunnerStatsRepository,
			in_filestorage: FileStorage,
			dataloader: DataLoader,
			in_path: str,
			tmp_path: str = "./",
			ma_window: int = 10,
			shuffle_order: bool = True,
			raise_exception: bool = False,
			exception_exceptions: typing.List[typing.Type] = None,
			temperatures: typing.Tuple[float, ...] = (1.0,),
			aggregate_alphas: typing.Tuple[float, ...] = (None,),
			horizon_mode: bool = False,
			horizon_bounds: typing.List[float] = None,
			horizon_h: float = None,
			horizon_max_depth: int = None,
			checkpointed: bool = False,
			horizon_model_class: typing.Type = HorizonModel,
			horizon_extra_args: typing.Dict[str, typing.Any] = None
	):
		self.__in_filestorage = in_filestorage
		self.__in_path = in_path
		self.__repository = repository
		self.__tmp_path = tmp_path
		self.__dataloader = dataloader
		self.__ma_window = ma_window
		self.__shuffle_order = shuffle_order
		self.__raise_exception = raise_exception
		self.__temperatures = temperatures
		self.__aggregate_alphas = aggregate_alphas
		self.__junk = set([])
		self.__checkpointed = checkpointed

		if exception_exceptions is None:
			exception_exceptions = []
		self.__exception_exceptions = exception_exceptions
		self.__loss_functions = self.get_evaluation_loss_functions()
		self.__blacklist_repo: RSBlacklistRepository = ResearchProvider.provide_rs_blacklist_repository(rs_repo=repository)

		self.__horizon_mode = horizon_mode
		self.__horizon_bounds = horizon_bounds
		self.__horizon_h = horizon_h
		self.__horizon_max_depth = horizon_max_depth
		self.__horizon_model_class = horizon_model_class
		self.__horizon_extra_args = horizon_extra_args if horizon_extra_args is not None else {}
		if self.__horizon_mode:
			assert self.__horizon_bounds is not None and self.__horizon_h is not None

	def __generate_tmp_path(self, ex=MODEL_SAVE_EXTENSION):
		return os.path.join(self.__tmp_path, f"{datetime.now().timestamp()}.{ex}")

	def __evaluate_model_loss(self, model: nn.Module, cls_loss_fn: SpinozaLoss) -> float:
		print(f"[+]Evaluating Model with {cls_loss_fn} loss...")
		evaluator = ModelEvaluator(
			dataloader=self.__dataloader,
			cls_loss_fn=cls_loss_fn,
		)
		loss = evaluator.evaluate(model)
		return loss[0]

	def __sync_model_losses_size(self, stat: RunnerStats):
		if len(stat.model_losses) < len(self.__loss_functions):
			stat.model_losses = tuple(stat.model_losses) + tuple([0.0,] * (len(self.__loss_functions) - len(stat.model_losses)))

	@staticmethod
	def get_evaluation_loss_functions() -> typing.List[SpinozaLoss]:
		return [
			ProximalMaskedLoss3.load(
				path=os.path.join(Config.RES_DIR, "losses/pml3_1.json"),
				bounds=DataPrepUtils.apply_bound_epsilon(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND),
				multi_channel=True
			),
			ProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				multi_channel=True
			)
		]

	def __evaluate_model(self, model: nn.Module, current_losses) -> typing.Tuple[float, ...]:
		if not self.__checkpointed:
			return tuple([
				self.__evaluate_model_loss(
					model,
					loss
				) if current_losses is None or current_losses[i] == 0.0 else current_losses[i]
				for i, loss in enumerate(self.__loss_functions)
			])

		current_losses = [0 for _ in self.__loss_functions] if current_losses is None else current_losses
		losses = list(current_losses).copy()
		i = current_losses.index(0.0)
		losses[i] = self.__evaluate_model_loss(model, self.__loss_functions[i])
		return tuple(losses)

	@staticmethod
	def __prepare_model(model: nn.Module) -> nn.Module:
		return model

	def __clean_junk(self):
		print(f"[+]Cleaning Junk...")
		for path in self.__junk:
			os.system(f"rm {os.path.abspath(path)}")
		self.__junk = set([])

	@staticmethod
	def __generate_id(file_path: str, temperature: float, aggregate_alpha) -> str:
		id = os.path.basename(file_path).replace(MODEL_SAVE_EXTENSION, "")
		if temperature != 1.0:
			id = f"{id}-(T={temperature})"
		if aggregate_alpha is not None:
			id = f"{id}-(A={aggregate_alpha})"
		return id

	@CacheDecorators.cached_method()
	def __download_model(self, path: str):
		print(f"[+]Downloading...")
		local_path = self.__generate_tmp_path()
		self.__in_filestorage.download(path, local_path)
		return local_path

	def _process_model(self, path: str, temperature: float, alpha: float):
		print(f"[+]Processing {path}(T={temperature}, A={alpha})...")

		stat = self.__repository.retrieve(self.__generate_id(path, temperature, alpha))
		if stat is not None:
			self.__sync_model_losses_size(stat)

		current_losses = stat.model_losses if stat is not None else None

		local_path = self.__download_model(path)
		model = ModelHandler.load(local_path)
		if self.__horizon_mode and isinstance(model, HorizonModel):
			Logger.warning(f"Stripping HorizonModel...")
			model = model.model

		if alpha is not None:
			model = AggregateModel(
				model=model,
				bounds=self.__horizon_bounds,
				a=alpha,
				softmax=True
			)

		model = TemperatureScalingModel(
			model,
			temperature=temperature
		)

		if self.__horizon_mode:
			model = self.__horizon_model_class(
				model=model,
				h=self.__horizon_h,
				bounds=self.__horizon_bounds,
				max_depth=self.__horizon_max_depth,
				**self.__horizon_extra_args
			)

		if current_losses is not None and False not in [loss == 0 for loss in current_losses]:
			current_losses = None

		print(f"[+]Evaluating...")
		losses = self.__evaluate_model(model, current_losses)
		id = self.__generate_id(path, temperature, alpha)

		stats = self.__repository.retrieve(id)
		if stats is None:
			print("[+]Creating...")
			stats = RunnerStats2(
				id=id,
				model_name=os.path.basename(path),
				temperature=temperature
			)
			stats.model_losses = losses
		else:
			print("[+]Updating...")
			stats.model_losses = losses
		self.__repository.store(stats)
		self.__junk.add(local_path)

	def __is_processed(self, file_path: str, temperature: float, aggregate_alpha: float) -> bool:
		stat_id = self.__generate_id(file_path, temperature, aggregate_alpha)

		stat = self.__repository.retrieve(stat_id)

		if stat is not None:
			self.__sync_model_losses_size(stat)

		return self.__blacklist_repo.is_blacklisted(stat_id) or (stat is not None and 0.0 not in stat.model_losses)

	def __is_all_complete(self,) -> bool:
		files = self.__in_filestorage.listdir(self.__in_path)
		for file in files:
			for temperature in self.__temperatures:
				for alpha in self.__aggregate_alphas:
					if not self.__is_processed(file, temperature, alpha):
						return False

		return True

	def start(self, replace_existing: bool = False):
		files = self.__in_filestorage.listdir(self.__in_path)
		if self.__shuffle_order:
			print("[+]Shuffling Files")
			random.shuffle(files)
		for i, file in enumerate(files):
			for temperature in self.__temperatures:
				for alpha in self.__aggregate_alphas:
					try:
						if self.__is_processed(file, temperature, alpha) and not replace_existing:
							print(f"[+]Skipping {file}(T={temperature}). Already Processed")
							continue
						self._process_model(file, temperature, alpha)
					except (FileNotFoundException, ) as ex:
						print(f"[-]Error Occurred processing {file}\n{ex}")
						if (
								self.__raise_exception or
								True in [isinstance(ex, exception_class) for exception_class in self.__exception_exceptions]
						):
							raise ex

			print(f"{(i+1)*100/len(files) :.2f}", end="\r")

		if not self.__is_all_complete():
			Logger.info(f"Found incomplete files. Restarting...")
			self.start(replace_existing)

		self.__clean_junk()
