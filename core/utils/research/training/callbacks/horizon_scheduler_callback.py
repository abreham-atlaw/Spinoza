from lib.utils.logger import Logger
from .callback import Callback
from ...data.prepare.smoothing_algorithm.lass.model.model import LassHorizonModel
from ...model.model.utils import HorizonModel, AbstractHorizonModel


class HorizonSchedulerCallback(Callback):

	def __init__(self, epochs: int, start: float = 0, end: float = 0.9, step: int = 1):
		super().__init__()
		self.__epochs = epochs
		self.__start = start
		self.__end = end
		self.__step = (self.__end - self.__start) / self.__epochs
		self.__epoch_step = step
		Logger.info(f"HorizonSchedulerCallback: Using step {self.__step}@{self.__epoch_step} epochs")

	def on_epoch_end(self, model: HorizonModel, epoch: int, losses, logs=None):
		if not isinstance(model, (HorizonModel, LassHorizonModel, AbstractHorizonModel)):
			Logger.warning(f"HorizonSchedulerCallback can only be used with HorizonModel. Got {type(model)}")
			return
		epoch = (epoch // self.__epoch_step) * self.__epoch_step
		h = min(self.__end, self.__start + self.__step * epoch)
		Logger.info(f"HorizonSchedulerCallback: Setting horizon from {model.h} to {h}")
		model.set_h(h)
