import os.path
import unittest

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from core import Config
from core.utils.research.model.model.utils import AggregateModel, WrappedModel, TemperatureScalingModel
from lib.utils.torch_utils.model_handler import ModelHandler


class AggregateModelTest(unittest.TestCase):


	def setUp(self):
		self.raw_model =ModelHandler.load(os.path.join(Config.BASE_DIR, "/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-training-cnn-0-it-97-tot.0_1.zip")).eval()
		self.wrapped_model = WrappedModel(
			TemperatureScalingModel(
				temperature=1.0,
				model=self.raw_model
			),
			seq_len=128
		)
		self.model = AggregateModel(
			model=self.wrapped_model,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			a=[0.99/5],
			temperature=1e-5
		).eval()
		self.softmax_model = AggregateModel(
			model=self.raw_model,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			a=0.98/3,
			temperature=1e-5,
			softmax=True
		)


		self.X = torch.from_numpy(np.load(os.path.join(Config.BASE_DIR, "temp/Data/simulation_simulator_data/13/train/X/1770301912.635284.npy")).astype(np.float32))

	def test_aggregate(self):
		with torch.no_grad():
			y = self.wrapped_model(self.X)[...,0 , :-1]
			y_hat = self.model.aggregate(y)

		for i in np.random.randint(0, self.X.shape[0], 5):
			plt.figure()
			plt.plot(y[i], label="Raw")
			plt.plot(y_hat[i], label="Aggregated")
			plt.legend()

		plt.show()

	def test_forward(self):

		with torch.no_grad():
			y = self.wrapped_model(self.X)[..., :-1]
			y_hat = self.model(self.X)[..., :-1]

		print(torch.sum(y_hat[~(torch.eq(torch.sum(y_hat, dim=-1), 1))], dim=-1))

		self.assertTrue(torch.all(torch.sum(y_hat, dim=-1) == 1.0))

		if len(y.shape) == 2:
			y, y_hat = [torch.unsqueeze(arr, dim=1) for arr in [y, y_hat]]


		for i in np.random.randint(0, self.X.shape[0], 5):
			plt.figure()

			for j in range(y.shape[1]):
				plt.subplot(2, 2, j+1)
				plt.plot(y[i, j], label="Raw")
				plt.plot(y_hat[i, j], label="Aggregated")
				plt.ylim([0, 1])
				plt.legend()

		plt.show()

	def test_aggregate_consistency(self):
		with torch.no_grad():
			y_hat = torch.stack([
				self.model(self.X)[..., :-1]
				for _ in range(10)
			], dim=0)

		for j in np.random.randint(0, y_hat.shape[1], 3):
			plt.figure()

			for i in range(y_hat.shape[0]):
				plt.plot(y_hat[i, j, 0], label=f"Call: {i}")
			plt.legend()
		plt.show()

	def test_softmax(self):

		softmax = nn.Softmax(dim=-1)
		with torch.no_grad():
			y = self.model(self.X)[..., :-1]
			y_hat = softmax(self.softmax_model(self.X)[..., :-1])

		if len(y.shape) == 2:
			y, y_hat = [torch.unsqueeze(arr, dim=1) for arr in [y, y_hat]]


		for i in np.random.randint(0, self.X.shape[0], 5):
			plt.figure()

			for j in range(y.shape[1]):
				plt.subplot(2, 2, j+1)
				plt.plot(y[i, j], label="Non-Softmax")
				plt.plot(y_hat[i, j], label="Softmax")
				plt.ylim([0, 1])
				plt.legend()

		plt.show()

	def test_custom(self):

		import numpy as np
		import re

		def parse_numpy_print(s: str) -> np.ndarray:
			"""
			Parse a NumPy array string (as printed by numpy) back into a numpy.ndarray.
			Works with multi-line arrays and missing commas.
			"""
			# Remove outer whitespace and quotes if present
			s = s.strip().strip('"').strip("'")

			# Count rows by detecting top-level bracket groups
			# Split rows like: [1 2 3] [4 5 6]
			row_strings = re.findall(r'\[([^\[\]]+)\]', s)

			if not row_strings:
				raise ValueError("No array rows found in the provided string.")

			# Convert each row to floats
			rows = []
			for row in row_strings:
				# Extract all float-like numbers (handles scientific notation too)
				numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', row)
				rows.append([float(n) for n in numbers])

			return np.array(rows, dtype=float)

		input_str = """[[1.02486 1.02548 1.02536 1.02598 1.02659 1.02595 1.02627 1.02608 1.02563
  1.026   1.02576 1.02434 1.02526 1.02525 1.02521 1.02511 1.02674 1.02748
  1.02882 1.02972 1.02866 1.02816 1.02882 1.02894 1.02989 1.02982 1.03032
  1.02968 1.03058 1.03028 1.03066 1.03072 1.03076 1.0306  1.03044 1.03019
  1.03016 1.03018 1.03008 1.03002 1.02966 1.02998 1.03014 1.02993 1.03015
  1.03034 1.03012 1.02962 1.02957 1.02918 1.03028 1.03056 1.02969 1.03016
  1.03098 1.03098 1.03078 1.03094 1.03022 1.03054 1.02987 1.03015 1.0336
  1.03378 1.03318 1.0323  1.03028 1.02918 1.02769 1.02856 1.02849 1.02913
  1.02925 1.02897 1.02934 1.02958 1.02968 1.02908 1.02893 1.02918 1.02925
  1.02939 1.0293  1.02988 1.02956 1.02968 1.029   1.02886 1.02874 1.0286
  1.02931 1.02873 1.02861 1.02824 1.02836 1.02844 1.02941 1.03002 1.0296
  1.02955 1.02924 1.0291  1.02868 1.02931 1.0286  1.02874 1.0286  1.0272
  1.0269  1.02736 1.02734 1.02706 1.02946 1.03    1.0311  1.03046 1.02995
  1.03098 1.02984 1.02946 1.0303  1.0304  1.02996 1.0302  1.02986 1.03018
  1.03025 1.03016]
 [1.02458 1.02457 1.02516 1.025   1.0259  1.02542 1.02536 1.02568 1.0252
  1.02562 1.0256  1.02422 1.02395 1.02445 1.02422 1.02436 1.02497 1.02573
  1.0263  1.02836 1.02829 1.02791 1.02816 1.02843 1.02878 1.02965 1.02976
  1.02964 1.02968 1.03008 1.03028 1.03042 1.0305  1.03055 1.03025 1.02997
  1.02998 1.03001 1.0298  1.02992 1.02953 1.02964 1.02988 1.02976 1.0299
  1.02985 1.02994 1.02918 1.02952 1.02867 1.02917 1.03016 1.02964 1.02943
  1.03    1.03049 1.03008 1.03041 1.02996 1.02997 1.02935 1.02961 1.03002
  1.03284 1.03284 1.03098 1.02989 1.0287  1.02583 1.02748 1.02842 1.02824
  1.02882 1.02893 1.02896 1.02904 1.02948 1.02906 1.02882 1.02875 1.02912
  1.02908 1.0293  1.02903 1.02847 1.02912 1.02876 1.02848 1.02854 1.02847
  1.0285  1.02869 1.02851 1.02822 1.02809 1.02823 1.02841 1.02934 1.02947
  1.0288  1.02888 1.02895 1.02817 1.0286  1.0284  1.02852 1.02855 1.02692
  1.02641 1.02608 1.02692 1.02679 1.02696 1.02883 1.02943 1.0286  1.02976
  1.02983 1.02944 1.0294  1.02934 1.02989 1.0298  1.02974 1.02968 1.02981
  1.02969 1.03008]
 [1.02518 1.02663 1.02624 1.02648 1.0275  1.0272  1.02666 1.02652 1.02634
  1.02632 1.02616 1.02606 1.02543 1.02552 1.0273  1.02649 1.02749 1.02783
  1.02899 1.03012 1.03044 1.02918 1.02885 1.02903 1.03003 1.02998 1.03056
  1.03045 1.03086 1.03062 1.0308  1.03084 1.03086 1.03084 1.0306  1.03066
  1.0303  1.03049 1.0302  1.03015 1.03013 1.03015 1.03023 1.03023 1.03019
  1.0304  1.03054 1.03015 1.02988 1.02976 1.03081 1.03174 1.03062 1.03072
  1.03132 1.03142 1.0311  1.03125 1.03099 1.03066 1.0306  1.03024 1.03548
  1.03508 1.03452 1.03341 1.0331  1.03079 1.02922 1.02921 1.02933 1.02924
  1.02932 1.02936 1.02946 1.0296  1.02991 1.02968 1.0291  1.02928 1.02953
  1.02944 1.02964 1.03    1.02998 1.02976 1.02983 1.02907 1.02922 1.02885
  1.0294  1.02934 1.02884 1.02864 1.0285  1.02894 1.02942 1.03016 1.03005
  1.02989 1.0297  1.0299  1.02914 1.02963 1.0294  1.02906 1.0292  1.02866
  1.02748 1.02819 1.02852 1.02768 1.0295  1.03059 1.03117 1.03117 1.03085
  1.03152 1.03106 1.03029 1.03044 1.03078 1.03056 1.03039 1.03019 1.03028
  1.03038 1.03038]
 [1.02488 1.02578 1.02536 1.02587 1.02678 1.02566 1.02597 1.02607 1.02551
  1.02618 1.02594 1.02433 1.02514 1.02514 1.02538 1.02504 1.02704 1.02768
  1.02836 1.02986 1.02862 1.02811 1.02873 1.02889 1.02994 1.02966 1.03031
  1.02975 1.03055 1.03028 1.03065 1.03066 1.03076 1.0307  1.03026 1.03021
  1.03018 1.03017 1.03003 1.03005 1.0296  1.03003 1.0302  1.02994 1.03008
  1.03034 1.0301  1.02966 1.02957 1.02908 1.03046 1.03036 1.02993 1.03022
  1.031   1.03084 1.03083 1.03074 1.03022 1.0304  1.0297  1.03017 1.03324
  1.0341  1.03339 1.0319  1.03016 1.02941 1.02768 1.02876 1.02856 1.0291
  1.02918 1.02902 1.02938 1.02958 1.02969 1.02912 1.02897 1.02919 1.02926
  1.0294  1.02948 1.02993 1.02976 1.02964 1.02896 1.0288  1.02873 1.02853
  1.0293  1.0291  1.02866 1.02826 1.02831 1.02851 1.02933 1.03004 1.0296
  1.02958 1.02916 1.02919 1.02856 1.02928 1.02867 1.02872 1.02872 1.02718
  1.02693 1.02773 1.02754 1.02704 1.02893 1.02972 1.03061 1.03097 1.03005
  1.03102 1.0298  1.02944 1.0304  1.03032 1.02999 1.03026 1.02989 1.03024
  1.03029 1.03016]] """

		x = torch.from_numpy(np.expand_dims(parse_numpy_print(input_str).astype(np.float32), axis=0))

		softmax = nn.Softmax(dim=-1)
		with torch.no_grad():
			y = self.model(x)
			y_hat = self.wrapped_model(x)
		print(torch.sum(y[..., :-1], dim=-1))
		print(torch.sum(y_hat[..., :-1], dim=-1))






