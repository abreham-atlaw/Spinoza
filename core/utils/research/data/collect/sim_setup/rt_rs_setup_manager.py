from core import Config
from core.utils.misc.sim_trading.setup import SetupManager
from core.utils.research.utils.model_evaluator import ModelEvaluator
from core.utils.resman import ResourceRepository
from lib.utils.file_storage import FileStorage
from lib.utils.logger import Logger
from .rs_setup_manager import RSSetupManager
from ..runner_stats import RunnerStats
from ..runner_stats_repository import RunnerStatsRepository


class RealTimeRSSetupManager(RSSetupManager):

	def __init__(
			self,
			rs_repo: RunnerStatsRepository,
			fs: FileStorage,
			model_evaluator: ModelEvaluator,
			accounts_repo: ResourceRepository
	):

		super().__init__(
			times_repo=None,
			rs_repo=rs_repo,
			fs=fs,
			model_evaluator=model_evaluator
		)
		self.__accounts_repo = accounts_repo
		self.__allocation_map = {}

	def _init_setup_manager(self) -> SetupManager:
		return SetupManager(setup_accounts=False)

	def __allocate_account(self):
		Logger.info(f"Allocating Oanda Account...")
		account = self.__accounts_repo.allocate()
		Logger.success(f"Allocated account: {account}")

		Config.OANDA_TRADING_ACCOUNT_ID = account.id
		return account

	def __unallocate_account(self, account):
		Logger.info(f"Unallocating Oanda Account...")
		self.__accounts_repo.release(account)
		Logger.success(f"Unallocated account: {account}")

	def _allocate_extra(self, stat: RunnerStats):
		self.__allocation_map[stat.id] = self.__allocate_account()
		self._setup_manager.setup()

	def _finish_extra(self, stat: RunnerStats):
		if stat.id not in self.__allocation_map:
			Logger.warning(f"Account not allocated for {stat.id}")
		else:
			self.__unallocate_account(self.__allocation_map[stat.id])
