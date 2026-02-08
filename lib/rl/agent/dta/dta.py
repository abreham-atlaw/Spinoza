import typing
from typing import *
from abc import ABC, abstractmethod

import numpy as np

import os

from lib.rl.environment import ModelBasedState
from . import Model
from .. import ModelBasedAgent
from ..utils.state_predictor import StatePredictor

TRANSITION_MODEL_FILE_NAME = "transition_model.h5"


class DNNTransitionAgent(ModelBasedAgent, ABC):

	def __init__(
			self,
			*args,
			batch_update=True,
			update_batch_size=100,
			clear_update_batch=True,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self._enable_batch_update = batch_update
		self._update_batch_size = update_batch_size
		if self._enable_batch_update:
			self._update_batch = ([], [])
		self._clear_update_batch = clear_update_batch

		self._predictor: StatePredictor = self._init_predictor()
		self._validation_dataset = ([], [])

	@abstractmethod
	def _init_predictor(self) -> StatePredictor:
		pass

	@abstractmethod
	def _prepare_dta_output(
			self, initial_state: List[ModelBasedState], output: np.ndarray, final_state: List[ModelBasedState]
	) -> List[float]:
		pass

	@abstractmethod
	def _prepare_dta_train_output(
			self, initial_state: List[ModelBasedState], action: List[Any], final_state: List[ModelBasedState]
	) -> np.ndarray:
		pass

	def __get_unique_rows(self, array: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
		hashes = np.array([hash(x.tobytes()) for x in array])
		hashes, indices, inverse = np.unique(hashes, axis=0, return_inverse=True, return_index=True)
		return array[indices], inverse

	def _call_predictor(self, initial_states: List[ModelBasedState], action: List[Any], final_states: List[ModelBasedState]) -> List[float]:
		return self._predictor.predict(initial_states, action, final_states=final_states)

	def _get_expected_transition_probability_distribution(
			self, initial_states: List[ModelBasedState], action: List[Any], final_states: List[ModelBasedState]
	) -> List[float]:

		prediction = self._call_predictor(initial_states, action, final_states)

		prediction = self._prepare_dta_output(initial_states, prediction, final_states)

		return list(prediction)

	def _update_model(self, batch=None):
		pass

	def __add_validation_set(self, X, y):
		self._validation_dataset[0].extend(X)
		self._validation_dataset[1].extend(y)

	def _update_transition_probability(self, initial_state: ModelBasedState, action, final_state: ModelBasedState):
		new_batch = (
			self._predictor.prepare_input([initial_state], [action], [final_state]),
			self._prepare_dta_train_output([initial_state], [action], [final_state])
		)

		if not self._enable_batch_update:
			self._update_model(new_batch)
			return

		self._update_batch[0].extend(new_batch[0])
		self._update_batch[1].extend(new_batch[1])

		if len(self._update_batch[0]) >= self._update_batch_size:
			self._update_model()
			self.__add_validation_set(*self._update_batch)
			if self._clear_update_batch:
				self._update_batch = ([], [])

	def get_configs(self):
		configs = super().get_configs()
		configs.update({
			"batch_update": self._enable_batch_update,
			"update_batch_size": self._update_batch_size
		})
		return configs

	def save(self, location):
		super().save(location)
		self.__transition_model.save(os.path.join(location, TRANSITION_MODEL_FILE_NAME))

	@staticmethod
	def load_configs(location):
		configs = ModelBasedAgent.load_configs(location)
		configs["model"] = keras.models.load_model(os.path.join(location, TRANSITION_MODEL_FILE_NAME))
		return configs
