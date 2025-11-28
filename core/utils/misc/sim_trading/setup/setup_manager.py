import os
from datetime import datetime

from core import Config
from core.di import ServiceProvider
from core.utils.misc.sim_trading.setup.requests import CreateAccountRequest
from lib.network.oanda import OandaNetworkClient
from lib.network.oanda.data.models import AccountSummary
from lib.utils.logger import Logger


class SetupManager:

	def __init__(
			self,
			setup_dirs: bool = True,
			setup_accounts: bool = True
	):
		self.__client = ServiceProvider.provide_oanda_client()
		self.__setup_dirs = setup_dirs
		self.__setup_accounts = setup_accounts

	@staticmethod
	def setup_dirs():
		Logger.info(f"Setting up directories...")
		for path in [Config.DUMP_CANDLESTICKS_PATH, Config.AGENT_DUMP_NODES_PATH, Config.UPDATE_SAVE_PATH]:
			Logger.info(f"Creating directory \"{path}\"...")
			os.makedirs(path, exist_ok=True)
		Logger.success(f"Directories Successfully Setup!")

	def setup_account(self, start_time: datetime) -> AccountSummary:
		Logger.info(f"Setting up account...")
		summary: AccountSummary = self.__client.execute(CreateAccountRequest(
			start_time=start_time,
			delta_multiplier=Config.OANDA_SIM_DELTA_MULTIPLIER,
			margin_rate=Config.OANDA_SIM_MARGIN_RATE,
			alias=Config.OANDA_SIM_ALIAS,
			balance=Config.OANDA_SIM_BALANCE
		))

		Config.OANDA_TRADING_ACCOUNT_ID = summary.id
		Logger.success(f"Account Successfully Setup!")
		return summary

	def setup(
		self,
		start_time: datetime = None
	):
		if self.__setup_dirs:
			self.setup_dirs()

		if self.__setup_accounts:
			summary = self.setup_account(start_time)
			return summary

		return None