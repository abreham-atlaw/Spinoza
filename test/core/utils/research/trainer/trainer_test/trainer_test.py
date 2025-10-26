import os
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from core import Config
from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.losses import CrossEntropyLoss, MeanSquaredErrorLoss, ReverseMAWeightLoss, ProximalMaskedLoss2, \
	ProximalMaskedPenaltyLoss2
from core.utils.research.model.layers import Indicators, DynamicLayerNorm, DynamicBatchNorm, MinMaxNorm, Axis, \
	LayerStack, Identity
from core.utils.research.model.model.cnn.bridge_block import BridgeBlock
from core.utils.research.model.model.cnn.cnn2 import CNN2
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.cnn.resnet.resnet_block import ResNetBlock
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Transformer, DecoderBlock, TransformerEmbeddingBlock, \
	TransformerBlock
from core.utils.research.model.model.utils import HorizonModel
from core.utils.research.training.callbacks.horizon_scheduler_callback import HorizonSchedulerCallback
from core.utils.research.training.trainer import Trainer
from core.utils.research.utils.model_migration.cnn_to_cnn2_migrator import CNNToCNN2Migrator
from lib.utils.torch_utils.model_handler import ModelHandler


class SineWaveDataset(Dataset):
	def __init__(self, total_samples, sequence_length, resolution=500):
		self.total_samples = total_samples
		self.sequence_length = sequence_length
		self.resolution = resolution
		self.data = np.sin(np.linspace(0, 2 * np.pi * resolution, total_samples + sequence_length))

	def __len__(self):
		return self.total_samples

	def shuffle(self):
		np.random.shuffle(self.data)

	def __getitem__(self, idx):
		start = idx
		end = start + self.sequence_length
		sequence = self.data[start:end]
		next_value = np.expand_dims(self.data[end], axis=0)
		return torch.tensor(sequence, dtype=torch.float32), torch.tensor(next_value, dtype=torch.float32)


class TrainerTest(unittest.TestCase):

	def _get_root_dirs(self):
		return [
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/simulation_simulator_data/02/train"
		], [
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/simulation_simulator_data/02/test"
		]

	def __init_dataloader(self):

		train_dirs, test_dirs = self._get_root_dirs()

		dataset = BaseDataset(
			train_dirs,
			check_file_sizes=True,
			load_weights=False,
			out_dtypes=self.np_dtype,
			num_files=2
		)
		dataloader = DataLoader(dataset, batch_size=64)

		test_dataset = BaseDataset(
			test_dirs,
			check_file_sizes=True,
			load_weights=False,
			out_dtypes=self.np_dtype,
			num_files=2
		)
		test_dataloader = DataLoader(test_dataset, batch_size=64)

		return dataloader, test_dataloader

	def _get_vocab_size(self):
		return len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1

	def _get_sequence_length(self):
		return 1024

	def _get_extra_len(self):
		return 124

	def _create_model(self):
		return self._create_horizon_model()

	def _create_cnn(self):
		CHANNELS = [128 for _ in range(4)]
		EXTRA_LEN = 124
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = self._get_vocab_size()
		POOL_SIZES = [(0, 0.5, 3) for _ in CHANNELS]
		DROPOUT_RATE = 0
		ACTIVATION = [nn.LeakyReLU(), nn.Identity(), nn.Identity(), nn.LeakyReLU()]
		BLOCK_SIZE = 1024 + EXTRA_LEN
		PADDING = 0
		LINEAR_COLLAPSE = True
		AVG_POOL = True
		NORM = [True] + [True for _ in CHANNELS[1:]]
		STRIDE = 2
		INPUT_DROPOUT = 0.2
		INPUT_NORM = False
		COLLAPSE_AVG_POOL = False
		LR = 1e-4

		POSITIONAL_ENCODING = True
		POSITIONAL_ENCODING_NORM = True
		INDICATORS_DELTA = True
		INDICATORS_SO = []
		INDICATORS_RSI = []
		INDICATORS_IDENTITIES = 4

		USE_CHANNEL_FFN = False
		CHANNEL_FFN_LAYERS = [CHANNELS[-1] for _ in range(4)]
		CHANNEL_FFN_DROPOUT = 0.1
		CHANNEL_FFN_ACTIVATION = nn.LeakyReLU()
		CHANNEL_FFN_NORM = [False] + [False for _ in CHANNEL_FFN_LAYERS[:-1]]
		CHANNEL_FFN_INIT = None

		USE_FF = True
		FF_LINEAR_LAYERS = [256 for _ in range(4)] + [VOCAB_SIZE + 1]
		FF_LINEAR_ACTIVATION = nn.LeakyReLU()
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [DynamicBatchNorm()] + [MinMaxNorm() for _ in FF_LINEAR_LAYERS[:-1]]
		FF_NORM_LEARNABLE = False
		FF_DROPOUT = 0.12

		if USE_FF:
			ff = LinearModel(
				dropout_rate=FF_DROPOUT,
				layer_sizes=FF_LINEAR_LAYERS,
				hidden_activation=FF_LINEAR_ACTIVATION,
				init_fn=FF_LINEAR_INIT,
				norm=FF_LINEAR_NORM,
				norm_learnable=FF_NORM_LEARNABLE
			)
		else:
			ff = None

		if USE_CHANNEL_FFN:
			channel_ffn = LinearModel(
				dropout_rate=CHANNEL_FFN_DROPOUT,
				layer_sizes=CHANNEL_FFN_LAYERS,
				hidden_activation=CHANNEL_FFN_ACTIVATION,
				init_fn=CHANNEL_FFN_INIT,
				norm=CHANNEL_FFN_NORM
			)
		else:
			channel_ffn = None

		indicators = Indicators(
			delta=INDICATORS_DELTA,
			so=INDICATORS_SO,
			rsi=INDICATORS_RSI,
			identities=INDICATORS_IDENTITIES,
		)

		model = CNN(
			extra_len=EXTRA_LEN,
			conv_channels=CHANNELS,
			kernel_sizes=KERNEL_SIZES,
			hidden_activation=ACTIVATION,
			pool_sizes=POOL_SIZES,
			dropout_rate=DROPOUT_RATE,
			padding=PADDING,
			avg_pool=AVG_POOL,
			linear_collapse=LINEAR_COLLAPSE,
			norm=NORM,
			ff_block=ff,
			indicators=indicators,
			input_size=BLOCK_SIZE,
			positional_encoding=POSITIONAL_ENCODING,
			norm_positional_encoding=POSITIONAL_ENCODING_NORM,
			stride=STRIDE,
			channel_ffn=channel_ffn,
			input_dropout=INPUT_DROPOUT,
			input_norm=INPUT_NORM,
			collapse_avg_pool=COLLAPSE_AVG_POOL,
		)
		return model

	def _create_cnn2(self):

		EXTRA_LEN = 124
		BLOCK_SIZE = 256 + EXTRA_LEN
		VOCAB_SIZE = len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1

		HORIZON_MODE = True
		HORIZON_BOUNDS = Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		HORIZON_RANGE = (0.0, 0.99)
		HORIZON_EPOCHS = 40
		HORIZON_STEP = 5
		HORIZON_MAX_DEPTH = 50

		CHANNELS = [128 for _ in range(8)]
		KERNEL_SIZES = [3 for _ in CHANNELS]
		POOL_SIZES = [(0, 0.5, 3, 2) for _ in CHANNELS[:2]] + [0 for _ in CHANNELS[2:]]
		DROPOUT_RATE = [0] * 2 + [0.2] * len(CHANNELS[2:])
		ACTIVATION = [nn.Identity() for _ in range(2)] + [nn.LeakyReLU() for _ in CHANNELS[2:-2]] + [nn.Identity() for _
																									 in range(2)]
		PADDING = 0
		LINEAR_COLLAPSE = True
		AVG_POOL = True
		NORM = [DynamicLayerNorm(elementwise_affine=False)] + [Identity() for _ in CHANNELS[1:]]

		INDICATORS_DELTA = True
		INDICATORS_SO = []
		INDICATORS_RSI = []

		INPUT_NORM = nn.Identity()

		DROPOUT_BRIDGE = 0.7

		USE_FF = True
		FF_LINEAR_LAYERS = [128] * 4 + [VOCAB_SIZE + 1]
		FF_LINEAR_ACTIVATION = [nn.Identity() for _ in range(2)] + [nn.LeakyReLU() for _ in FF_LINEAR_LAYERS[2:-1]]
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [nn.Identity() for _ in FF_LINEAR_LAYERS[:]]
		FF_DROPOUT = [0.3 for _ in FF_LINEAR_LAYERS[:-1]]

		indicators = Indicators(
			delta=INDICATORS_DELTA,
			so=INDICATORS_SO,
			rsi=INDICATORS_RSI
		)

		model = CNN2(
			extra_len=EXTRA_LEN,
			input_size=BLOCK_SIZE,

			embedding_block=EmbeddingBlock(
				indicators=indicators,
				input_norm=INPUT_NORM
			),

			cnn_block=CNNBlock(
				input_channels=indicators.indicators_len,
				conv_channels=CHANNELS,
				kernel_sizes=KERNEL_SIZES,
				pool_sizes=POOL_SIZES,
				hidden_activation=ACTIVATION,
				dropout_rate=DROPOUT_RATE,
				norm=NORM,
				padding=PADDING
			),

			collapse_block=CollapseBlock(
				dropout=DROPOUT_BRIDGE,
				ff_block=LinearModel(
					dropout_rate=FF_DROPOUT,
					layer_sizes=FF_LINEAR_LAYERS,
					hidden_activation=FF_LINEAR_ACTIVATION,
					init_fn=FF_LINEAR_INIT,
					norm=FF_LINEAR_NORM
				)
			)

		)

		if HORIZON_MODE:
			model = HorizonModel(
				model=model,
				bounds=HORIZON_BOUNDS,
				h=HORIZON_RANGE[0],
				max_depth=HORIZON_MAX_DEPTH
			)
		return model

	def _create_transformer(self):

		EXTRA_LEN = 124
		INPUT_SIZE = 1024 + EXTRA_LEN
		VOCAB_SIZE = 431
		EMBEDDING_SIZE = 32

		# EMBEDDING BLOCK
		EMBEDDING_CB_CHANNELS = [8]*2 + [EMBEDDING_SIZE]
		EMBEDDING_CB_KERNELS = [3]*len(EMBEDDING_CB_CHANNELS)
		EMBEDDING_CB_POOL_SIZES = [0] * len(EMBEDDING_CB_CHANNELS)
		EMBEDDING_CB_DROPOUTS = [0] * len(EMBEDDING_CB_CHANNELS)
		EMBEDDING_CB_NORM = [False] * len(EMBEDDING_CB_CHANNELS)
		EMBEDDING_CB_HIDDEN_ACTIVATION = nn.PReLU()
		EMBEDDING_POSITIONAL_ENCODING = True

		# DECODER_BLOCK
		DECODER_NORM_1 = DynamicLayerNorm()
		DECODER_NORM_2 = DynamicLayerNorm()
		DECODER_FF_LAYERS = [64, EMBEDDING_SIZE]

		INDICATORS_DELTA = True

		DECORDER_NUM_HEADS = 4

		# ENCODER_BLOCK
		ENCODER_NUM_HEADS = 4
		ENCODER_NORM_1 = DynamicLayerNorm()
		ENCODER_NORM_2 = DynamicLayerNorm()
		ENCODER_FF_LAYERS = [128, EMBEDDING_SIZE]


		# COLLAPSE BLOCK
		COLLAPSE_FF_LAYERS = [16] * 2 + [VOCAB_SIZE + 1]
		COLLAPSE_FF_DROPOUTS = [0] * (len(COLLAPSE_FF_LAYERS) - 1)
		COLLAPSE_FF_ACTIVATION = nn.PReLU()
		COLLAPSE_FF_NORM = [False] * len(COLLAPSE_FF_LAYERS)

		indicators = Indicators(
			delta=INDICATORS_DELTA,
		)

		return Transformer(
			input_size=INPUT_SIZE,
			extra_len=EXTRA_LEN,
			transformer_block=TransformerBlock(

				transformer_embedding_block=TransformerEmbeddingBlock(
					positional_encoding=EMBEDDING_POSITIONAL_ENCODING,
					embedding_block=EmbeddingBlock(
						indicators=indicators
					),
					cnn_block=CNNBlock(
						input_channels=indicators.indicators_len,
						conv_channels=EMBEDDING_CB_CHANNELS,
						kernel_sizes=EMBEDDING_CB_KERNELS,
						pool_sizes=EMBEDDING_CB_POOL_SIZES,
						dropout_rate=EMBEDDING_CB_DROPOUTS,
						norm=EMBEDDING_CB_NORM,
						hidden_activation=EMBEDDING_CB_HIDDEN_ACTIVATION
					)
				),

				encoder_block=DecoderBlock(
					num_heads=ENCODER_NUM_HEADS,
					norm_1=ENCODER_NORM_1,
					norm_2=ENCODER_NORM_2,
					ff_block=LinearModel(
						layer_sizes=ENCODER_FF_LAYERS
					)
				),

				decoder_block=DecoderBlock(
					num_heads=DECORDER_NUM_HEADS,
					norm_1=DECODER_NORM_1,
					norm_2=DECODER_NORM_2,
					ff_block=LinearModel(
						layer_sizes=DECODER_FF_LAYERS
					),
				)
			),

			collapse_block=CollapseBlock(
				ff_block=LinearModel(
					layer_sizes=COLLAPSE_FF_LAYERS,
					hidden_activation=COLLAPSE_FF_ACTIVATION,
					dropout_rate=COLLAPSE_FF_DROPOUTS,
					norm=COLLAPSE_FF_NORM
				),
			)
		)

	def _create_horizon_model(self):
		return HorizonModel(
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			model=self._create_cnn2(),
			h=0.9
		)

	def _create_losses(self):
		return (
			ProximalMaskedLoss2(e=2.0, w=1.0, n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1, weighted_sample=False),
			MeanSquaredErrorLoss(weighted_sample=False)
		)

	def __init_trainer(self, model):

		callbacks = [
			HorizonSchedulerCallback(
				epochs=10,
				start=0,
				end=0.5,
				step=2
			)
		]
		trainer = Trainer(model, callbacks=callbacks, skip_nan=True, dtype=self.dtype)
		trainer.cls_loss_function, trainer.reg_loss_function = self._create_losses()
		trainer.optimizer = Adam(trainer.model.parameters())
		return trainer

	def _get_reg_loss_only(self) -> bool:
		return False

	def setUp(self):
		self.dtype = torch.float32
		self.np_dtype = np.float32
		self.model = self._create_model()
		self.dataloader, self.test_dataloader = self.__init_dataloader()
		self.trainer = self.__init_trainer(self.model)

	def test_train(self):

		SAVE_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra.zip"

		self.trainer.train(
			self.dataloader,
			val_dataloader=self.test_dataloader,
			epochs=1,
			progress=True,
			reg_loss_only=self._get_reg_loss_only()
		)

		model = self.trainer.model
		if isinstance(model, HorizonModel):
			model = model.model

		ModelHandler.save(model, SAVE_PATH)

		for X, y, w in self.test_dataloader:
			break

		loaded_model = ModelHandler.load(SAVE_PATH).to(self.dtype)
		loaded_model.eval()

		original_y = model(X)

		loaded_y = loaded_model(X)

		self.assertTrue(torch.allclose(original_y, loaded_y))

		self.trainer.validate(self.test_dataloader)

	def test_validate(self):
		self.trainer.validate(self.test_dataloader)
