import os
import typing

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core import Config
from core.utils.research.data.load import BaseDataset
from core.utils.research.losses import ProximalMaskedLoss, SpinozaLoss, MeanSquaredErrorLoss
from core.utils.research.training.trainer import Trainer
from lib.utils.logger import Logger


class ModelEvaluator:

	def __init__(
			self,
			cls_loss_fn: SpinozaLoss,
			dataloader: DataLoader = None,
			data_path: typing.Union[str, typing.List[str]] = None,
			batch_size: int = 32,
			reg_loss_fn: SpinozaLoss = None,
			dtype = torch.float32,
			np_dtype = np.float32
	):
		assert (dataloader is not None) or (data_path is not None)

		if isinstance(data_path, str):
			data_path = [data_path]
		self.__cls_loss_fn = cls_loss_fn
		if reg_loss_fn is None:
			reg_loss_fn = MeanSquaredErrorLoss()
		self.__reg_loss_fn = reg_loss_fn
		self.__data_path, self.__batch_size = data_path, batch_size
		self.__dataloader = dataloader
		self.__dtype, self.__np_dtype = dtype, np_dtype

	def __init_dataloader(self, paths: typing.List[str], batch_size: int):
		Logger.info(f"Initializing Dataloader...")
		dataset = BaseDataset(
			root_dirs=paths,
			out_dtypes=self.__np_dtype,
			check_file_sizes=True
		)
		dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
		return dataloader

	@property
	def dataloader(self) -> DataLoader:
		if self.__dataloader is None:
			self.__dataloader = self.__init_dataloader(self.__data_path, self.__batch_size)
		return self.__dataloader

	def __init_trainer(self, model):
		trainer = Trainer(model, dtype=self.__dtype)
		trainer.cls_loss_function = self.__cls_loss_fn
		trainer.reg_loss_function = self.__reg_loss_fn
		return trainer

	def _evaluate(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module):
		trainer = self.__init_trainer(model)
		losses = trainer.validate(dataloader)
		return losses

	def evaluate(self, model) -> typing.Tuple[float, float, float]:
		model.eval()

		losses = self._evaluate(model, self.dataloader, self.__cls_loss_fn)

		return losses

	def __call__(self, model) -> typing.Tuple[float, float, float]:
		return self.evaluate(model)
