import os

from torch import nn

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers import SmoothedChannelDropout, \
	EncoderNoiseInjectionLayer
from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers.lass3.transformer import \
	DecodedEncoderDropout, EncoderChannelDropout, EncoderDropout, DecoderPadEmbedding, EncoderVerticalShift
from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model import LassHorizonModel
from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model.lass3 import Lass3HorizonModel
from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model.lass3.transformer import Lass3Transformer, \
	Lass3DecoderBlock, CrossAttentionBlock, Lass3DeltaModel
from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model.lass3.transformer.lass3_transformer_input_block import \
	Lass3TransformerInputBlock
from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model.lass5 import Lass3To5Model, \
	Lass3ExecutorLass5Model
from core.utils.research.losses import MeanSquaredErrorLoss
from core.utils.research.model.layers import DynamicLayerNorm, DynamicBatchNorm, Indicators
from core.utils.research.model.model.cnn.cnn2 import CNN2
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import TransformerEmbeddingBlock, DecoderBlock
from .trainer_test import TrainerTest


class LassTrainerTest(TrainerTest):

	def _get_root_dirs(self):
		return [
			os.path.join(Config.BASE_DIR, "temp/Data/lass/7/train")
		], [
			os.path.join(Config.BASE_DIR, "temp/Data/lass/7/test")
		]

	def _create_losses(self):
		return (
			None,
			MeanSquaredErrorLoss(weighted_sample=False)
		)

	def __create_cnn2(self):
		INPUT_CHANNELS = 2

		EMBEDDING_SIZE = 32

		CHANNELS = [EMBEDDING_SIZE for _ in range(4)]
		EXTRA_LEN = self._get_extra_len()
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = self._get_vocab_size()
		POOL_SIZES = [0 for _ in CHANNELS]
		DROPOUT_RATE = [0 for _ in CHANNELS]
		ACTIVATION = [nn.Identity(), nn.Identity(), nn.LeakyReLU(), nn.Identity()]
		BLOCK_SIZE = self._get_sequence_length() + EXTRA_LEN
		PADDING = 0
		NORM = [DynamicLayerNorm()] + [nn.Identity() for _ in CHANNELS[1:]]

		INDICATORS_DELTA = [1]
		INDICATORS_SO = []
		INDICATORS_RSI = []
		INPUT_NORM = DynamicLayerNorm()
		SMOOTHING_DROPOUT = SmoothedChannelDropout(batch_dropout=0.5, depth_dropout=0.5)

		COLLAPSE_INPUT_NORM = DynamicBatchNorm()
		DROPOUT_BRIDGE = 0.2
		COLLAPSE_GLOBAL_AVG_POOL = True
		COLLAPSE_EXTRA_MODE = False

		FF_LINEAR_LAYERS = [64, 16] + [VOCAB_SIZE + 1]
		FF_LINEAR_ACTIVATION = [nn.Identity(), nn.LeakyReLU()]
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [nn.Identity() for _ in FF_LINEAR_LAYERS[:]]
		FF_DROPOUT = 0

		indicators = Indicators(
			delta=INDICATORS_DELTA,
			so=INDICATORS_SO,
			rsi=INDICATORS_RSI,
			input_channels=INPUT_CHANNELS
		)

		return CNN2(
			extra_len=EXTRA_LEN,
			input_size=(None, INPUT_CHANNELS, BLOCK_SIZE) if INPUT_CHANNELS > 1 else BLOCK_SIZE,

			embedding_block=EmbeddingBlock(
				indicators=indicators,
				input_norm=INPUT_NORM,
				input_dropout=SMOOTHING_DROPOUT
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
				extra_mode=COLLAPSE_EXTRA_MODE,
				dropout=DROPOUT_BRIDGE,
				input_norm=COLLAPSE_INPUT_NORM,
				global_avg_pool=COLLAPSE_GLOBAL_AVG_POOL,
				ff_block=LinearModel(
					dropout_rate=FF_DROPOUT,
					layer_sizes=FF_LINEAR_LAYERS,
					hidden_activation=FF_LINEAR_ACTIVATION,
					init_fn=FF_LINEAR_INIT,
					norm=FF_LINEAR_NORM
				)
			)

		)

	@staticmethod
	def __create_lass3_transformer():

		BLOCK_SIZE = 32
		EMBEDDING_SIZE = 4
		NUM_HEADS = 2
		VOCAB_SIZE = 1

		# HORIZON
		HORIZON_MODE = True
		HORIZON_RANGE = (0.0, 0.9)
		HORIZON_EPOCHS = 70
		HORIZON_STEP = 5
		HORIZON_MAX_DEPTH = None

		# INPUT_BLOCK
		DECODER_PAD_EMBEDDING = DecoderPadEmbedding()
		ENCODER_VERTICAL_SHIFT = 1.0

		# ENCODER EMBEDDING BLOCK
		ENCODER_EMBEDDING_INDICATORS_DELTA = [1]
		ENCODER_EMBEDDING_PADDING = nn.ReflectionPad1d((1, 0))
		ENCODER_EMBEDDING_CB_CHANNELS = [EMBEDDING_SIZE] * 8
		ENCODER_EMBEDDING_CB_KERNELS = [3] * len(ENCODER_EMBEDDING_CB_CHANNELS)
		ENCODER_EMBEDDING_CB_POOL_SIZES = [0] * len(ENCODER_EMBEDDING_CB_CHANNELS)
		ENCODER_EMBEDDING_CB_DROPOUTS = [0] * len(ENCODER_EMBEDDING_CB_CHANNELS)
		ENCODER_EMBEDDING_CB_NORM = [DynamicLayerNorm() for _ in range(len(ENCODER_EMBEDDING_CB_CHANNELS))]
		ENCODER_EMBEDDING_CB_HIDDEN_ACTIVATION = [nn.Identity() for _ in ENCODER_EMBEDDING_CB_CHANNELS]
		ENCODER_EMBEDDING_CB_PADDING = [nn.ReflectionPad1d(padding=(e // 2)) for e in ENCODER_EMBEDDING_CB_KERNELS]
		ENCODER_EMBEDDING_POSITIONAL_ENCODING = True

		# ENCODER BLOCK
		ENCODER_NUM_HEADS = NUM_HEADS
		ENCODER_NORM_1 = DynamicLayerNorm()
		ENCODER_NORM_2 = DynamicLayerNorm()
		ENCODER_FF_LAYERS = [EMBEDDING_SIZE * 2, EMBEDDING_SIZE]

		# DECODER EMBEDDING BLOCK
		DECODER_EMBEDDING_INDICATORS_DELTA = []
		DECODER_EMBEDDING_CB_CHANNELS = [EMBEDDING_SIZE] * 1
		DECODER_EMBEDDING_CB_KERNELS = [1] * len(DECODER_EMBEDDING_CB_CHANNELS)
		DECODER_EMBEDDING_CB_POOL_SIZES = [0] * len(DECODER_EMBEDDING_CB_CHANNELS)
		DECODER_EMBEDDING_CB_DROPOUTS = [0] * len(DECODER_EMBEDDING_CB_CHANNELS)
		DECODER_EMBEDDING_CB_NORM = [nn.Identity() for _ in range(len(DECODER_EMBEDDING_CB_CHANNELS))]
		DECODER_EMBEDDING_CB_HIDDEN_ACTIVATION = [nn.Identity() for _ in DECODER_EMBEDDING_CB_CHANNELS]
		DECODER_EMBEDDING_POSITIONAL_ENCODING = True

		# DECODER SELF ATTENTION BLOCK
		DECODER_SELF_ATTENTION_NUM_HEADS = NUM_HEADS
		DECODER_SELF_ATTENTION_NORM_1 = DynamicLayerNorm()

		# DECODER CROSS ATTENTION BLOCK
		DECODER_CROSS_ATTENTION_NUM_HEADS = NUM_HEADS
		DECODER_CROSS_ATTENTION_NORM_1 = DynamicLayerNorm()
		DECODER_CROSS_ATTENTION_NORM_2 = DynamicLayerNorm()
		DECODER_CROSS_ATTENTION_FF_LAYERS = [EMBEDDING_SIZE * 2, EMBEDDING_SIZE]

		# COLLAPSE BLOCK
		COLLAPSE_BRIDGE_DROPOUT = 0
		COLLAPSE_INPUT_NORM = nn.Identity()
		COLLAPSE_GLOBAL_AVG_POOL = False
		COLLAPSE_FF_LINEAR_LAYERS = [EMBEDDING_SIZE * 8, EMBEDDING_SIZE * 4, VOCAB_SIZE]
		COLLAPSE_FF_LINEAR_ACTIVATION = [nn.Identity() for _ in COLLAPSE_FF_LINEAR_LAYERS]
		COLLAPSE_FF_LINEAR_NORM = [nn.Identity() for _ in COLLAPSE_FF_LINEAR_LAYERS]
		COLLAPSE_FF_LINEAR_DROPOUT = [0] * (len(COLLAPSE_FF_LINEAR_LAYERS) - 1)

		encoder_indicators = Indicators(
			delta=ENCODER_EMBEDDING_INDICATORS_DELTA
		)

		decoder_indicators = Indicators(
			delta=DECODER_EMBEDDING_INDICATORS_DELTA
		)

		model = Lass3Transformer(
			block_size=BLOCK_SIZE,

			input_block=Lass3TransformerInputBlock(
				decoder_prep=DecoderPadEmbedding(),
				encoder_prep=EncoderVerticalShift(ENCODER_VERTICAL_SHIFT)
			),

			encoder_embedding_block=TransformerEmbeddingBlock(
				positional_encoding=True,
				embedding_block=EmbeddingBlock(
					indicators=encoder_indicators,
					padding=ENCODER_EMBEDDING_PADDING
				),
				cnn_block=CNNBlock(
					input_channels=encoder_indicators.indicators_len,
					conv_channels=ENCODER_EMBEDDING_CB_CHANNELS,
					kernel_sizes=ENCODER_EMBEDDING_CB_KERNELS,
					pool_sizes=ENCODER_EMBEDDING_CB_POOL_SIZES,
					dropout_rate=ENCODER_EMBEDDING_CB_DROPOUTS,
					norm=ENCODER_EMBEDDING_CB_NORM,
					hidden_activation=ENCODER_EMBEDDING_CB_HIDDEN_ACTIVATION,
					padding=ENCODER_EMBEDDING_CB_PADDING
				)
			),

			decoder_embedding_block=TransformerEmbeddingBlock(
				positional_encoding=True,
				embedding_block=EmbeddingBlock(
					indicators=decoder_indicators
				),
				cnn_block=CNNBlock(
					input_channels=decoder_indicators.indicators_len,
					conv_channels=DECODER_EMBEDDING_CB_CHANNELS,
					kernel_sizes=DECODER_EMBEDDING_CB_KERNELS,
					pool_sizes=DECODER_EMBEDDING_CB_POOL_SIZES,
					dropout_rate=DECODER_EMBEDDING_CB_DROPOUTS,
					norm=DECODER_EMBEDDING_CB_NORM,
					hidden_activation=DECODER_EMBEDDING_CB_HIDDEN_ACTIVATION,
				)
			),

			encoder_block=DecoderBlock(
				embedding_last=False,
				num_heads=ENCODER_NUM_HEADS,
				norm_1=ENCODER_NORM_1,
				norm_2=ENCODER_NORM_2,
				ff_block=LinearModel(ENCODER_FF_LAYERS)
			),

			decoder_block=Lass3DecoderBlock(
				self_attention_block=DecoderBlock(
					embedding_last=False,
					num_heads=DECODER_SELF_ATTENTION_NUM_HEADS,
					norm_1=DECODER_SELF_ATTENTION_NORM_1
				),
				cross_attention_block=CrossAttentionBlock(
					num_heads=DECODER_CROSS_ATTENTION_NUM_HEADS,
					norm_1=DECODER_CROSS_ATTENTION_NORM_1,
					norm_2=DECODER_CROSS_ATTENTION_NORM_2,
					ff_block=LinearModel(DECODER_CROSS_ATTENTION_FF_LAYERS)
				),
				transpose_output=True
			),

			collapse_block=CollapseBlock(
				extra_mode=False,
				dropout=COLLAPSE_BRIDGE_DROPOUT,
				input_norm=COLLAPSE_INPUT_NORM,
				global_avg_pool=COLLAPSE_GLOBAL_AVG_POOL,
				ff_block=LinearModel(
					dropout_rate=COLLAPSE_FF_LINEAR_DROPOUT,
					layer_sizes=COLLAPSE_FF_LINEAR_LAYERS,
					norm=COLLAPSE_FF_LINEAR_NORM,
					hidden_activation=COLLAPSE_FF_LINEAR_ACTIVATION
				)
			)
		)

		if HORIZON_MODE:
			model = Lass3HorizonModel(
				h=HORIZON_RANGE[0],
				model=model,
				max_depth=HORIZON_MAX_DEPTH
			)
		return model

	def __create_lass5_model(self):
		model = self.__create_lass3_transformer()
		return Lass3ExecutorLass5Model(padding=2, block_size=32, model=model)

	def _create_model(self):
		# return LassHorizonModel(
		# 	h=0.5,
		# 	model=self.__create_cnn2()
		# )
		# return Lass3HorizonModel(
		# 	h=0.5,
		# 	model=self.__create_lass3_transformer(),
		# 	max_depth=5
		# )
		return self.__create_lass3_transformer()
		# return self.__create_lass5_model()

	def _get_sequence_length(self):
		return 32

	def _get_extra_len(self):
		return 0

	def _get_vocab_size(self) -> int:
		return 0

	def _get_reg_loss_only(self) -> bool:
		return True

	def test_train(self):
		super().test_train()
