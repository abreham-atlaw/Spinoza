import os
import typing
from datetime import datetime

import numpy as np
import torch

from core.utils.research.losses import SpinozaLoss
from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.logger import Logger


class ConfidencePreparer:

	def __init__(
			self,
			data_path: str,
			export_path: str,
			model: SpinozaModule,
			loss: SpinozaLoss,
			x_encoder_dir_name = "X_Encoder",
			x_decoder_dir_name = "X_Decoder",
			y_dir_name = "Y_Encoder",
			x_input_dir_name = "X",
			y_input_dir_name = "y",
			y_extra_len: int = 1,
	):
		self.__x_path = os.path.join(data_path, x_input_dir_name)
		self.__y_path = os.path.join(data_path, y_input_dir_name)
		self.__x_encoder_dir = x_encoder_dir_name
		self.__x_decoder_dir = x_decoder_dir_name
		self.__y_dir = y_dir_name
		self.__export_path = export_path
		self.__model = model.eval()
		self.__loss = loss
		self.__y_extra_len = y_extra_len
		self.__files = self.__get_filenames(self.__x_path)

		if loss.collapsed:
			Logger.warning(f"Received collapsed loss.")

	def __setup_dirs(self):
		for dir_name in [self.__x_encoder_dir, self.__x_decoder_dir, self.__y_dir]:
			path = os.path.join(self.__export_path, dir_name)
			Logger.info(f"Setting up {path}...")
			os.makedirs(path, exist_ok=True)

	@staticmethod
	def __get_filenames(path: str) -> typing.List[str]:
		return list(filter(
			lambda filename: filename.endswith(".npy"),
			os.listdir(path)
		))

	def __process_batch(self, x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			y_hat = self.__model(x)

		loss = self.__loss(y_hat[..., :y_hat.shape[-1] - self.__y_extra_len], y[..., :y_hat.shape[-1] - self.__y_extra_len])
		return 1/loss

	@staticmethod
	def __load_file(path: str) -> torch.Tensor:
		return torch.from_numpy(np.load(path).astype(np.float32))

	def __save_batch(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor, y: torch.Tensor):
		filename = f"{datetime.now().timestamp()}.npy"
		for arr, dir_name in zip([x_encoder, x_decoder, y], [self.__x_encoder_dir, self.__x_decoder_dir, self.__y_dir]):
			path = os.path.join(self.__export_path, dir_name, filename)
			np.save(path, arr.numpy())

	def __process_file(self, filename: str):
		x, y = [self.__load_file(os.path.join(container, filename)) for container in [self.__x_path, self.__y_path]]
		c = self.__process_batch(x, y)

		self.__save_batch(x, y, c)

	def __main(self):
		for i, filename in enumerate( self.__files):
			self.__process_file(filename)
			Logger.info(f"Processed: {i+1}/{len(self.__files)}", end="\r")

	def start(self):
		self.__setup_dirs()
		self.__main()
