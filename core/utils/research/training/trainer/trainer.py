
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import uuid
import typing
import os

from core import Config
from core.di import ResearchProvider
from core.utils.research.training.callbacks import Callback
from core.utils.research.training.data.state import TrainingState
from core.utils.research.training.trackers.tracker import TorchTracker
from lib.utils.logger import Logger
from lib.utils.torch_utils.model_handler import ModelHandler
from .device import Device
from ...data.load.dataset import BaseDataset
from ...losses import SpinozaLoss

try:
    from torch_xla.distributed import parallel_loader
    from torch_xla.core import xla_model
    import torch_xla.core.xla_model as xm

except ImportError:
    Logger.warning("XLA is not installed. Training using TPU will not be possible.")


class Trainer:

    def __init__(
            self,
            model,
            cls_loss_function: typing.Optional[SpinozaLoss]=None,
            reg_loss_function: typing.Optional[SpinozaLoss]=None,
            optimizer=None,
            callbacks: typing.List[Callback]=None,
            max_norm: typing.Optional[float] = None,
            clip_value: typing.Optional[float] = None,
            log_gradient_stats: bool = False,
            trackers: typing.List[TorchTracker] = None,
            dtype: torch.dtype = torch.float64,
            skip_nan: bool = True
    ):
        self.device = self.get_device()
        Logger.info(f"Using device: {self.device_type}")
        if torch.cuda.device_count() > 1:
            print("Found use", torch.cuda.device_count(), "GPUs.")
            model = torch.nn.DataParallel(model)
        if callbacks is None:
            callbacks = []
        self.__dtype = dtype
        self.model = self.__initialize_model(model)
        self.cls_loss_function = cls_loss_function
        self.reg_loss_function = reg_loss_function
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.__state: typing.Optional[TrainingState] = None
        self.__max_norm = max_norm
        self.__clip_value = clip_value
        self.__log_gradient_stats = log_gradient_stats
        self.__skip_nan = skip_nan
        self.__trackers = trackers if trackers is not None \
            else (ResearchProvider.provide_default_trackers(model_name=ModelHandler.generate_signature(model)))

    @staticmethod
    def get_device():
        try:
            import torch_xla
            from torch_xla.distributed import parallel_loader
            return xm.xla_device()
        except ImportError:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

    @property
    def device_type(self) -> int:
        if self.device.type == "xla":  # Check for TPU
            return Device.TPU
        elif self.device.type == "cuda":  # Check for GPU
            return Device.GPU
        else:
            return Device.CPU

    @property
    def state(self) -> typing.Optional[TrainingState]:
        return self.__state

    def summary(self):
        print("Model Summary")
        print("Layer Name" + "\t" * 7 + "Number of Parameters")
        print("=" * 100)
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            total_params += param
            print(name + "\t" * 3 + str(param))
        print("=" * 100)
        print(f"Total Params:{total_params}")

    @staticmethod
    def __get_default_state() -> TrainingState:
        return TrainingState(
            id=uuid.uuid4().hex,
            epoch=0,
            batch=0
        )

    def __initialize_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.__dtype)
        init_data = torch.rand((1,) + model.input_size[1:]).to(self.__dtype)
        model = model.to(torch.device("cpu"))
        model.eval()
        model(init_data)
        return model.to(self.device).to(self.__dtype)

    @staticmethod
    def __split_y(y: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return y[:, :-1], y[:, -1:]

    def __loss(self, y_hat: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        cls_y, reg_y = self.__split_y(y)
        cls_y_hat, reg_y_hat = self.__split_y(y_hat)

        cls_loss, reg_loss = torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)
        if self.cls_loss_function is not None:
            cls_loss = self.cls_loss_function(cls_y_hat, cls_y, w)
        if self.reg_loss_function is not None:
            reg_loss = self.reg_loss_function(reg_y_hat, reg_y, w)

        loss = cls_loss + reg_loss
        return cls_loss, reg_loss, loss

    @staticmethod
    def __format_loss(loss):
        return f"loss: {loss[2]}(cls: {loss[0]}, reg: {loss[1]})"

    def __prepare_dataloader(self, dataloader) -> DataLoader:
        if self.device_type == Device.TPU:
            dataloader = parallel_loader.MpDeviceLoader(dataloader, self.device)
        return dataloader

    def __optimizer_step(self):
        if self.device_type == Device.TPU:
            xm.optimizer_step(self.optimizer)
        else:
            self.optimizer.step()

    def __prepare_data(self, data: typing.Tuple[torch.Tensor,...]) -> typing.Tuple[torch.Tensor,...]:
        return [
            d.to(self.device).type(self.__dtype)
            for d in data
        ]

    def train(
            self,
            dataloader: DataLoader,
            val_dataloader=None,
            epochs: int = 1,
            progress: bool = False,
            shuffle=True,
            progress_interval=100,
            cls_loss_only=False,
            reg_loss_only=False,
            state: typing.Optional[TrainingState] = None
    ):
        if self.optimizer is None or (self.cls_loss_function is None and not reg_loss_only):
            raise ValueError("Model not setup(optimizer or loss function missing)")

        dataset: BaseDataset = dataloader.dataset

        dataloader = self.__prepare_dataloader(dataloader)
        if val_dataloader is not None:
            val_dataloader = self.__prepare_dataloader(val_dataloader)

        if state is None:
            state = self.__get_default_state()

        self.__state = state

        for callback in self.callbacks:
            callback.on_train_start(self.model)
        self.summary()

        if cls_loss_only:
            print("Training Classifier")

        if reg_loss_only:
            print("Training Regressor")

        train_losses = []
        val_losses = []
        for epoch in range(state.epoch, epochs):
            min_gradient = float('inf')
            max_gradient = float('-inf')
            state.epoch = epoch
            if shuffle:
                dataset.shuffle()
            for callback in self.callbacks:
                callback.on_epoch_start(self.model, epoch)
            self.model.train()
            self.model = self.model.to(self.device).to(self.__dtype)
            running_loss, running_size = torch.zeros((3,)), 0
            pbar = tqdm(dataloader) if progress else dataloader
            for i, data in enumerate(pbar):
                if i < state.batch:
                    continue
                state.batch = i
                for callback in self.callbacks:
                    callback.on_batch_start(self.model, i)

                X, y, w = self.__prepare_data(data)

                self.optimizer.zero_grad()
                y_hat = self.model(X)

                cls_loss, ref_loss, loss = self.__loss(y_hat, y, w)
                if cls_loss_only:
                    cls_loss.backward()
                elif reg_loss_only:
                    ref_loss.backward()
                else:
                    loss.backward()

                gradients = [param.grad.clone().detach() for param in self.model.parameters() if param.grad is not None]

                for tracker in self.__trackers:
                    tracker.on_batch_end(X, y, y_hat, self.model, loss, gradients, epoch, i)

                if self.__max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.__max_norm)

                if self.__clip_value is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.__clip_value)

                # self.optimizer.step()
                self.__optimizer_step()

                running_loss += torch.FloatTensor([l.item() for l in [cls_loss, ref_loss, loss]]) * X.shape[0]
                running_size += X.shape[0]
                if progress and i % progress_interval == 0:
                    pbar.set_description(f"Epoch {epoch + 1} {self.__format_loss(running_loss/running_size)}")
                for callback in self.callbacks:
                    callback.on_batch_end(self.model, i)
            state.batch = 0
            epoch_loss = (running_loss / running_size).tolist()
            print(f"Epoch {epoch + 1} completed, {self.__format_loss(epoch_loss)}")
            if self.__log_gradient_stats:
                print(f"Min gradient: {min_gradient}, Max gradient: {max_gradient}")

            train_losses.append(epoch_loss)
            losses = (epoch_loss,)

            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                val_losses.append(val_loss)
                print(f"Validation loss: {self.__format_loss(val_loss)}")
                losses = (epoch_loss, val_loss)

            for callback in self.callbacks:
                callback.on_epoch_end(self.model, epoch, losses)

        for callback in self.callbacks:
            callback.on_train_end(self.model)
        return train_losses, val_losses

    def validate(self, dataloader: DataLoader):
        Logger.info(f"Validating ...")
        self.model.eval()
        total_loss = torch.zeros((3,))
        total_size = 0
        with torch.no_grad():
            for data in dataloader:
                X, y, w = self.__prepare_data(data)
                y_hat = self.model(X)
                cls_loss, ref_loss, loss = self.__loss(y_hat, y, w)

                total_loss += torch.FloatTensor([l.item() for l in [cls_loss, ref_loss, loss]]) * X.shape[0]
                total_size += X.shape[0]
                if self.__skip_nan and torch.isnan(total_loss).any():
                    Logger.error("Nan value encountered. Skipping...")
                    break
        return (total_loss / total_size).tolist()
