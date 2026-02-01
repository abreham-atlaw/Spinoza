import os

from torch import nn

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model.lass3.transformer import CrossAttentionBlock
from core.utils.research.losses import MeanSquaredErrorLoss
from core.utils.research.model.layers import DynamicLayerNorm, PositionalEncoding, Indicators
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import TransformerEmbeddingBlock, DecoderBlock
from core.utils.research.utils.confidence.model.model.transformer import ConfidenceTransformer
from test.core.utils.research.trainer.trainer_test.trainer_test import TrainerTest


class ConfidenceTrainerTest(TrainerTest):

	def _create_losses(self):
		return (
			None,
			MeanSquaredErrorLoss(weighted_sample=False)
		)

	def _get_root_dirs(self):
		return [
			os.path.join(Config.BASE_DIR, "temp/Data/confidence_data/00")
		], [
			os.path.join(Config.BASE_DIR, "temp/Data/confidence_data/00")
		]

	@property
	def is_regression(self) -> bool:
		return True

	def __create_confidence_transformer(self):
		X_ENCODER_SIZE = 128
		INPUT_SHAPE = (3, X_ENCODER_SIZE+66)
		EMBEDDING_SIZE = 4
		NUM_HEADS = 2
		VOCAB_SIZE = 1

		# ENCODER EMBEDDING BLOCK
		ENCODER_EMBEDDING_INDICATORS_DELTA = []
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
		DECODER_EMBEDDING_CB_NORM = [DynamicLayerNorm() for _ in range(len(DECODER_EMBEDDING_CB_CHANNELS))]
		DECODER_EMBEDDING_CB_HIDDEN_ACTIVATION = [nn.Identity() for _ in DECODER_EMBEDDING_CB_CHANNELS]
		DECODER_EMBEDDING_CB_PADDING = [nn.ReflectionPad1d(padding=(e // 2)) for e in ENCODER_EMBEDDING_CB_KERNELS]

		# DECODER BLOCK
		DECODER_NUM_HEADS = NUM_HEADS
		DECODER_NORM_1 = DynamicLayerNorm()
		DECODER_NORM_2 = DynamicLayerNorm()
		DECODER_FF_LAYERS = [EMBEDDING_SIZE * 2, EMBEDDING_SIZE]

		# CROSS ATTENTION BLOCK
		CROSS_ATTENTION_NUM_HEADS = NUM_HEADS
		CROSS_ATTENTION_NORM_1 = DynamicLayerNorm()
		CROSS_ATTENTION_NORM_2 = DynamicLayerNorm()
		CROSS_ATTENTION_FF_LAYERS = [EMBEDDING_SIZE * 2, EMBEDDING_SIZE]

		# COLLAPSE BLOCK
		COLLAPSE_BRIDGE_DROPOUT = 0
		COLLAPSE_INPUT_NORM = nn.Identity()
		COLLAPSE_GLOBAL_AVG_POOL = False
		COLLAPSE_FF_LINEAR_LAYERS = [EMBEDDING_SIZE * 8, EMBEDDING_SIZE * 4, VOCAB_SIZE]
		COLLAPSE_FF_LINEAR_ACTIVATION = [nn.Identity() for _ in COLLAPSE_FF_LINEAR_LAYERS]
		COLLAPSE_FF_LINEAR_NORM = [nn.Identity() for _ in COLLAPSE_FF_LINEAR_LAYERS]
		COLLAPSE_FF_LINEAR_DROPOUT = [0] * (len(COLLAPSE_FF_LINEAR_LAYERS) - 1)

		encoder_indicators = Indicators(
			delta=ENCODER_EMBEDDING_INDICATORS_DELTA,
			input_channels=INPUT_SHAPE[0]
		)

		decoder_indicators = Indicators(
			delta=DECODER_EMBEDDING_INDICATORS_DELTA,
			input_channels=INPUT_SHAPE[0]
		)

		model = ConfidenceTransformer(
			input_shape=INPUT_SHAPE,
			encoder_embedding_block=TransformerEmbeddingBlock(
				positional_encoding=True,
				embedding_block=EmbeddingBlock(
					indicators=encoder_indicators,
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
				),
			),

			decoder_embedding_block=TransformerEmbeddingBlock(
				positional_encoding=True,
				embedding_block=EmbeddingBlock(
					indicators=decoder_indicators,
				),
				cnn_block=CNNBlock(
					input_channels=decoder_indicators.indicators_len,
					conv_channels=DECODER_EMBEDDING_CB_CHANNELS,
					kernel_sizes=DECODER_EMBEDDING_CB_KERNELS,
					pool_sizes=DECODER_EMBEDDING_CB_POOL_SIZES,
					dropout_rate=DECODER_EMBEDDING_CB_DROPOUTS,
					norm=DECODER_EMBEDDING_CB_NORM,
					hidden_activation=DECODER_EMBEDDING_CB_HIDDEN_ACTIVATION,
					padding=DECODER_EMBEDDING_CB_PADDING
				),
			),

			encoder_block=DecoderBlock(
				embedding_last=False,
				num_heads=ENCODER_NUM_HEADS,
				norm_1=ENCODER_NORM_1,
				norm_2=ENCODER_NORM_2,
				ff_block=LinearModel(ENCODER_FF_LAYERS)
			),

			decoder_block=DecoderBlock(
				embedding_last=False,
				num_heads=DECODER_NUM_HEADS,
				norm_1=DECODER_NORM_1,
				norm_2=DECODER_NORM_2,
				ff_block=LinearModel(DECODER_FF_LAYERS)
			),

			cross_attention_block=CrossAttentionBlock(
				embedding_last=False,
				num_heads=CROSS_ATTENTION_NUM_HEADS,
				norm_1=CROSS_ATTENTION_NORM_1,
				norm_2=CROSS_ATTENTION_NORM_2,
				ff_block=LinearModel(CROSS_ATTENTION_FF_LAYERS),
				add_decoder=True
			),

			collapse_block = CollapseBlock(
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
			),

		)

		return model

	def _create_model(self):
		return self.__create_confidence_transformer()

	def _get_reg_loss_only(self) -> bool:
		return True

	def test_train(self):
		super().test_train()