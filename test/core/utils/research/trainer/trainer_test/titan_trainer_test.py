import os

import torch.nn as nn

from core import Config
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.losses import ProximalMaskedLoss3, MeanSquaredErrorLoss
from core.utils.research.model.layers import AnchoredReturnsLayer, DynamicLayerNorm
from core.utils.research.model.model.cnn.bridge_block import BridgeBlock
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.titan import TitanModel, TitanInputBlock, TitanContextBlock, \
	TitanContextEmbeddingBlock, TitanTimeSeriesBlock
from core.utils.research.model.model.transformer import DecoderBlock, TransformerBlock, TransformerEmbeddingBlock
from lib.utils.fileio import load_json
from .trainer_test import TrainerTest


class TitanTrainerTest(TrainerTest):

	def _get_root_dirs(self):
		return [
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/simulation_simulator_data/12/train"
		], [
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/simulation_simulator_data/12/test"
		]

	def _create_losses(self):
		return (
			ProximalMaskedLoss3(
				bounds=DataPrepUtils.apply_bound_epsilon(load_json(os.path.join(Config.RES_DIR, "bounds/15.json"))),
				weighted_sample=False,
				multi_channel=True,
			),
			MeanSquaredErrorLoss(weighted_sample=False)
		)

	@staticmethod
	def __create_titan_model():

		CONTEXT_DATA_SIZE = 12
		EMBEDDING_SIZE = 64
		BLOCK_SIZE = 128 + CONTEXT_DATA_SIZE
		VOCAB_SIZE = len(load_json(os.path.join(Config.BASE_DIR, "res/bounds/15.json"))) + 1
		INSTRUMENTS = 2
		INPUT_CHANNELS = 4 * INSTRUMENTS
		OUTPUT_CHANNELS = 4 * INSTRUMENTS

		TIME_SERIES_CHANNELS = [EMBEDDING_SIZE for _ in range(8)]
		TIME_SERIES_KERNEL = [3] * len(TIME_SERIES_CHANNELS)
		TIME_SERIES_HIDDEN_ACTIVATIONS = [nn.Identity() for _ in TIME_SERIES_CHANNELS]


		TIME_SERIES_PREP_LAYERS = AnchoredReturnsLayer(
			anchored_channels=list(range(INPUT_CHANNELS)),
			anchor_channels=0
		)

		TIME_SERIES_HEADS = 4

		TIME_SERIES_PE_NORM = DynamicLayerNorm()

		TIME_SERIES_DECODER_HEADS = TIME_SERIES_HEADS
		TIME_SERIES_DECODER_NORM_1 = DynamicLayerNorm()
		TIME_SERIES_DECODER_NORM_2 = DynamicLayerNorm()
		TIME_SERIES_DECODER_FF_LAYERS = [EMBEDDING_SIZE * 2, EMBEDDING_SIZE]

		TIME_SERIES_ENCODER_HEADS = TIME_SERIES_HEADS
		TIME_SERIES_ENCODER_NORM_1 = DynamicLayerNorm()
		TIME_SERIES_ENCODER_NORM_2 = DynamicLayerNorm()
		TIME_SERIES_ENCODER_FF_LAYERS = [EMBEDDING_SIZE * 2, EMBEDDING_SIZE]

		DROPOUT_BRIDGE = 0

		TIME_SERIES_COLLAPSE_FF_LAYERS = [254, VOCAB_SIZE]

		TIME_SERIES_COLLAPSE_CHANNEL_FF_LAYERS = [32, 16, OUTPUT_CHANNELS]
		TIME_SERIES_COLLAPSE_FLATTEN = OUTPUT_CHANNELS == 1

		CONTEXT_INSTRUMENT_POSITIONS = (6, 11)
		CONTEXT_INSTRUMENT_EMBEDDING_SIZE = 10
		CONTEXT_FF_LAYERS = [128, 64]
		CONTEXT_TIME_SERIES_DECODER_BLOCK_HEADS = 4
		CONTEXT_TIME_SERIES_DECODER_NORM_1 = DynamicLayerNorm()
		CONTEXT_TIME_SERIES_DECODER_NORM_2 = DynamicLayerNorm()
		CONTEXT_TIME_SERIES_DECODER_FF_LAYERS = [EMBEDDING_SIZE*2, EMBEDDING_SIZE]
		CONTEXT_TIME_SERIES_FF_LAYERS = [128, 64, 1]
		CONTEXT_VALUE_HEAD_FF_LAYERS = [254, 128, 1]


		model = TitanModel(
			input_size=(INPUT_CHANNELS, BLOCK_SIZE),
			input_block=TitanInputBlock(context_data_size=CONTEXT_DATA_SIZE),

			context_block=TitanContextBlock(
				embedding_block=TitanContextEmbeddingBlock(
					instrument_positions=CONTEXT_INSTRUMENT_POSITIONS,
					instruments_vocab=INSTRUMENTS,
					embedding_size=CONTEXT_INSTRUMENT_EMBEDDING_SIZE
				),
				context_ffn=LinearModel(
					layer_sizes=CONTEXT_FF_LAYERS,
				),
				time_series_decoder_block=DecoderBlock(
					num_heads=CONTEXT_TIME_SERIES_DECODER_BLOCK_HEADS,
					norm_1=CONTEXT_TIME_SERIES_DECODER_NORM_1,
					norm_2=CONTEXT_TIME_SERIES_DECODER_NORM_2,
					ff_block=LinearModel(
						layer_sizes=CONTEXT_TIME_SERIES_DECODER_FF_LAYERS,
					)
				),
				time_series_ffn=LinearModel(
					layer_sizes=CONTEXT_TIME_SERIES_FF_LAYERS,
				),
				value_head=LinearModel(
					layer_sizes=CONTEXT_VALUE_HEAD_FF_LAYERS,
				)
			),

			time_series_block=TitanTimeSeriesBlock(
				embedding_block=EmbeddingBlock(
					prep_layer=TIME_SERIES_PREP_LAYERS,
				),
				cnn_block=CNNBlock(
					input_channels=INPUT_CHANNELS,
					conv_channels=TIME_SERIES_CHANNELS,
					kernel_sizes=TIME_SERIES_KERNEL,
					hidden_activation=TIME_SERIES_HIDDEN_ACTIVATIONS
				),
				bridge_block=BridgeBlock(
					transformer_block=TransformerBlock(
						transformer_embedding_block=TransformerEmbeddingBlock(
							pe_norm=TIME_SERIES_PE_NORM,
							positional_encoding=True
						),
						encoder_block=DecoderBlock(
							num_heads=TIME_SERIES_ENCODER_HEADS,
							norm_1=TIME_SERIES_ENCODER_NORM_1,
							norm_2=TIME_SERIES_ENCODER_NORM_2,
							ff_block=LinearModel(
								layer_sizes=TIME_SERIES_ENCODER_FF_LAYERS
							)
						),
						decoder_block=DecoderBlock(
							num_heads=TIME_SERIES_DECODER_HEADS,
							norm_1=TIME_SERIES_DECODER_NORM_1,
							norm_2=TIME_SERIES_DECODER_NORM_2,
							ff_block=LinearModel(
								layer_sizes=TIME_SERIES_DECODER_FF_LAYERS
							)
						)
					)
				),
				collapse_block=CollapseBlock(
					extra_mode=False,
					dropout=DROPOUT_BRIDGE,
					flatten=TIME_SERIES_COLLAPSE_FLATTEN,
					channel_ff_block=LinearModel(
						layer_sizes=TIME_SERIES_COLLAPSE_CHANNEL_FF_LAYERS
					),
					ff_block=LinearModel(
						layer_sizes=TIME_SERIES_COLLAPSE_FF_LAYERS,

					)
				)
			)
		)

		return model

	def _create_model(self):
		return self.__create_titan_model()

	def test_train(self):
		super().test_train()
