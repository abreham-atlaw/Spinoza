import typing

import numpy as np

import json
from datetime import datetime, timedelta

from lib.utils.logger import Logger


class SimulationTimesGenerator:

	def __init__(
			self,
			time_format: str = "%Y-%m-%d %H:%M:%S+00:00",
			random_mode: bool = False
	):
		self.__time_format = time_format
		self.__random_mode = random_mode

	def __generate_times(self, start_time: datetime, end_time: datetime, count: int) -> typing.List[datetime]:
		Logger.info(f"Generating {count} times between {start_time} and {end_time}")
		gap = int((end_time - start_time).total_seconds())

		if self.__random_mode:
			gaps = np.random.randint(0, gap, count)
		else:
			gaps = np.linspace(0, gap, count)

		dts = [start_time + timedelta(seconds=s) for s in gaps]

		return dts

	def __export_times(self, times: typing.List[datetime], export_path: str):
		Logger.info(f"Exporting times to {export_path}")
		with open(export_path, "w") as f:
			json.dump([dt.strftime(self.__time_format) for dt in times], f)

	def generate(
			self,
			start_time: datetime,
			end_time: datetime,
			count: int,
			export_path: str,
	):
		Logger.info(f"Generating...")
		times = self.__generate_times(start_time, end_time, count)
		self.__export_times(times, export_path)
		Logger.success(f"Successfully generated times!")
