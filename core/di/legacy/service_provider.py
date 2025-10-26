import torch
from pymongo import MongoClient

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm import Lass, SmoothingAlgorithm, MovingAverage
from core.utils.research.data.prepare.smoothing_algorithm.lass.executors import Lass3Executor
from core.utils.research.model.model.utils import AbstractHorizonModel
from core.utils.resman import ResourceRepository, MongoResourceRepository
from lib.dnn.layers import KalmanFilter
from lib.network.oanda import OandaNetworkClient
from lib.utils.file_storage import FileStorage, PCloudCombinedFileStorage
from lib.utils.torch_utils.model_handler import ModelHandler


class ServiceProvider:

	@staticmethod
	def provide_file_storage(path=None) -> FileStorage:
		if path is None:
			path = Config.PCLOUD_FOLDER
		return PCloudCombinedFileStorage(
			tokens=Config.PCLOUD_TOKENS,
			base_path=path
		)

	@staticmethod
	def provide_mongo_client() -> MongoClient:
		return MongoClient(Config.MONGODB_URL)

	@staticmethod
	def provide_resman(category: str) -> ResourceRepository:
		return MongoResourceRepository(category=category, mongo_client=ServiceProvider.provide_mongo_client())

	@staticmethod
	def provide_oanda_client() -> OandaNetworkClient:
		return OandaNetworkClient(
			url=Config.OANDA_TRADING_URL,
			token=Config.OANDA_TOKEN,
			account_id=Config.OANDA_TRADING_ACCOUNT_ID
		)

	@staticmethod
	def provide_lass(device: torch.device = None, fs_path: str = None) -> Lass:
		from core.utils.research.utils.model_utils import ModelUtils
		path = fs_path or Config.AGENT_LASS_MODEL_FS_PATH
		model = ModelUtils.load_from_fs(path).to(device=device)
		if isinstance(model, AbstractHorizonModel):
			model.set_h(0.0)
		return Lass(
			model=model,
			executor=Lass3Executor()
		)

	@staticmethod
	def provide_smoothing_algorithm() -> SmoothingAlgorithm:
		if Config.AGENT_USE_LASS:
			return ServiceProvider.provide_lass()

		if Config.AGENT_USE_KALMAN_FILTER:
			return KalmanFilter(
				alpha=Config.AGENT_KALMAN_ALPHA,
				beta=Config.AGENT_KALMAN_BETA
			)

		return MovingAverage(
			window_size=Config.AGENT_MA_WINDOW_SIZE
		)
