import typing
from typing import *

import cattr
import json

import numpy as np

T = TypeVar('T')

class Serializer(typing.Generic[T]):

	def __init__(self, output_class):
		self.__output_class = output_class

	def serialize(self, data: T) -> Dict:
		return cattr.unstructure(data)

	def serialize_json(self, data: T):
		return json.dumps(self.serialize(data))

	def deserialize(self, json_: Dict) -> T:
		return cattr.structure(json_, self.__output_class)

	def deserialize_json(self, json_: str):
		return self.deserialize(json.loads(json_))

	def serialize_many(self, instances: List[T]) -> List[Dict]:
		return [self.serialize(instance) for instance in instances]

	def deserialize_many(self, jsons: List[Dict]) -> List[object]:
		return [self.deserialize(json_) for json_ in jsons]


class NumpySerializer(Serializer):

	def __init__(self):
		super().__init__(np.ndarray)

	def serialize(self, data: np.ndarray) -> object:
		return data.tolist()

	def deserialize(self, json_: List) -> np.ndarray:
		return np.array(json_)
