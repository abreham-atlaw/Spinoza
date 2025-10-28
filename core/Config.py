import json

from pytz import timezone
import os
import random
from dataclasses import dataclass


@dataclass
class ModelConfig:
	id: str
	download: bool
	url: str
	path: str


BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

REMOTE_TRADER_URL = "http://localhost:8080/"
NETWORK_TRIES = 10

LOGGING = True
LOGGING_PID = True
LOGGING_CONSOLE = True
LOGGING_FILE_PATH = os.path.abspath("output.log")

MC_SERVER_PORT = 8000
MC_SERVER_URL = "http://127.0.0.1:%s" % (MC_SERVER_PORT,)

MIN_FREE_MEMORY = 2
MAX_PROCESSES = 6
RECURSION_DEPTH = 10000
NESTED_PROCESS = False
MAIN_PID = os.getpid()

DEFAULT_EPSILON = 1e-9

MONGODB_URL = "mongodb+srv://abreham:zYUir15jnOcrPqg1@cluster0.vn0ngnn.mongodb.net/?retryWrites=true&w=majority"

OPTIMIZER_PG_CONFIG = {
	"dsn": "postgres://ontiwpwwgbtgwp:8702c0dec88af3c49473d464bf44e8ad17419facfce764c8684ed540839fb8cb@ec2-34-194-100-156.compute-1.amazonaws.com:5432/dcs4e3sfc908fi",
	"sslmode": "require"
}

DROPBOX_API_TOKEN = "sl.BrBqqSAAzJsdIo9x1NG-z51HwPzc6yw0KCrbQ7hn76DvI0vItMEhXBDiqW-W4yGzHzCHL1RVBEeFeKEEdj7EIIjorHfLqOeM9KEhanIYOynP3XGiFRi_4RK3MGATuFHz50iZUFezZ09Uwd6Uk4LC3Oc"
DROPBOX_FOLDER = "/RForexTrader"

# PCLOUD_API_TOKEN = "wiaJJZfKQB7ZFJGbF2LjLo0nFysSdWCN0mklLXYV"  # main
# PCLOUD_API_TOKEN = "NbYQ47Z6O9B7ZsjyeitXJhUpmug9Cg4qS8m40yXYk"  # 0
# PCLOUD_API_TOKEN = "1Qbjq7ZIO9B7ZzfX5wncB5G7ebGSYi95oiVmjFkky" # 1

PCLOUD_TOKENS = [
	# "wKjUxVZfKQB7ZRhHVp3l82GbW3HxrluLXwRJzzjT7",  # abrishatlaw@gmail.com +
	"nC2uuVZ6O9B7ZjuXK6ioCWTXjAwmG3WvOFJkUu4PX",  # abrishatlaw@yahoo.com +
	# "QL3dHVZIO9B7Z3luRCUxoFefoSYQb1LRwkbOdFjoX",  # abreham.atlaw@yahoo.com -
	# "aCT8vkZxDks7ZYpOYIhqlahkcknASzvkHKLR8Ai3y",  # abrehamatlaw@outlook.com -
	"WoSiVVZHDks7Z7kGMSCexDu8dxeB1GClFzpDx9TOk",  # abreham.atlaw@outlook.com +
	# "bDBit7ZEWJs7ZvmomkVGYvr02Fd0DWd56ByQLbjLk",  # abreham-atlaw@outlook.com -
	"LrbFxZnvzs7ZBsXhKl0AR3plxs3YDDjYMy7x35pV",  # abreham_atlaw@outlook.com +
	"0q6NC7ZkqQs7Z7aVgEWJEiH7Lm9R1KWjbPpAi3b2X",  # abrehama@outlook.com +
	# "2WjwdXZiyRs7ZTBMoqYbCS2hvTbuzYbBP6XVkEByy",  # abreham.a@outlook.com +
	"zkfNekZPR4s7ZtNbg8OCFr75xqmxaiqVxxBix9UHk",  # abreham_a@outlook.com +
	# "TbW8dXZPays7ZaalmkXkXb40vpl0MxsA5Fp2TVsry",  # hiwotahab12@gmail.com +
	"2sgeXkZXe7s7Zx29adBJwFzV6PLXY3OOYsJNEFtok",  # abrehamatlaw321@gmail.com -
	"7zoKYXZktF97Z6gm3frhMpjjU9M08A58WgRda0PHX",  # abrehamalemu@outlook.com
	# "lmQOmkZWmKM7ZyodzaLpjx5S2KO1wNcPuIhrYzFUX"  # abreham-a@outlook.com
]

PCLOUD_API_TOKEN = "jfAYHkZfKQB7Zn0vw75zQgU82511XehVaVjc2zSRV"

PCLOUD_FOLDER = "/Apps/RTrader"
MODEL_PCLOUD_FOLDER = os.path.join(PCLOUD_FOLDER, "Models/10M/10MA")
CHECKPOINT_PCLOUD_FOLDER = os.path.join(PCLOUD_FOLDER, "Checkpoints/10M/10MA")

IMS_REMOTE_PATH = os.path.join(PCLOUD_FOLDER, "stats")
IMS_SYNC_SIZE = int(1e5)
IMS_TEMP_PATH = "/tmp/"

POLYGON_API_KEY = "1ijeQ0XUYNl1YMHy6Wl_5zEBtGbkipUP"

# ENVIRONMENT CONFIGS

# TRAINING ENVIRONMENT
HISTORICAL_TABLE_NAME = "test_currencies_history"
CURRENCIES_TABLE_NAME = "test_currencies"
TRADEABLE_PAIRS_TABLE_NAME = "test_tradeable_pairs"
DISTINCT_DATETIMES_TABLE_NAME = "test_distinct_datetimes"
"""
HISTORICAL_TABLE_NAME = "currencies_history"
CURRENCIES_TABLE_NAME = "currencies"
TRADEABLE_PAIRS_TABLE_NAME = "tradeable_pairs"
DISTINCT_DATETIMES_TABLE_NAME = "distinct_datetimes"
"""
DEFAULT_PG_CONFIG = {
	"host": "localhost",
	"database": "rtrader_db",
	"user": "rtrader_admin",
	"password": "4U7z7KJM"  # TODO
}

# LIVE ENVIRONMENT
OANDA_TOKEN = "4e3bc058fee3b2005e2a651081da881e-1cc2b5245cda5e61beb340aaf217c704"
OANDA_TRADING_URL = "https://api-fxpractice.oanda.com/v3"
OANDA_TRADING_ACCOUNT_ID = "101-001-19229086-002"
OANDA_TEST_ACCOUNT_ID = "101-001-19229086-002"

OANDA_SIM_DELTA_MULTIPLIER = 10
OANDA_SIM_MARGIN_RATE = 0.01
OANDA_SIM_BALANCE = 100
OANDA_SIM_ALIAS = "Sim Account 0"
OANDA_SIM_TIMES_PATH = os.path.join(BASE_DIR, "res/times/times-5.json")
OANDA_SIM_MODEL_IN_PATH = "/Apps/RTrader/"

DEFAULT_TIME_IN_FORCE = "FOK"
TIMEZONE = timezone("Africa/Addis_Ababa")

# AGENT CONFIGS
UPDATE_AGENT = True
UPDATE_EXPORT_BATCH_SIZE = 2
UPDATE_SAVE_PATH = os.path.join(BASE_DIR, "temp/Data/drmca_export")
UPDATE_TRAIN = False
MARKET_STATE_MEMORY = 128
MARKET_STATE_SMOOTHING = True
MARKET_STATE_GRANULARITY = "M30"
MARKET_STATE_USE_ANCHOR = False
DUMP_CANDLESTICKS_PATH = os.path.join(BASE_DIR, "temp/candlesticks/real")
TIME_PENALTY = 0
AGENT_TRADE_PENALTY = 0
AGENT_TRADE_SIZE_GAP = 70
AGENT_TRADE_MIN_SIZE = 50
AGENT_TRADE_SIZE_USE_PERCENTAGE = False
AGENT_DEPTH = 30  # TODO: DEPRECATED
AGENT_STATE_CHANGE_DELTA_MODEL_MODE = False
AGENT_MIN_PROBABILITY = 1e-6
AGENT_DISCOUNT_FACTOR = 1
AGENT_DISCOUNT_FUNCTION = None
AGENT_EXPLOIT_EXPLORE_TRADEOFF = 1
AGENT_UCT_EXPLORE_WEIGHT = 0.7
AGENT_LOGICAL_MCA = True
AGENT_FRICTION_TIME = 6
AGENT_STEP_TIME = (2 * 60) - AGENT_FRICTION_TIME
AGENT_USE_CUSTOM_RESOURCE_MANAGER = False
AGENT_MAX_INSTRUMENTS = 2
AGENT_USE_STATIC_INSTRUMENTS = True
AGENT_STATIC_INSTRUMENTS = [
	("AUD", "USD"),
	("USD", "ZAR")
]
AGENT_RANDOM_SEED = random.randint(0, 1000000)
AGENT_CURRENCY = "USD"
AGENT_CORE_PRICING = False
AGENT_COMMISSION_COST = 0.05  # IN AGENT_CURRENCY
AGENT_SPREAD_COST = 0.05  # IN AGENT_CURRENCY
AGENT_STM = True
AGENT_STM_THRESHOLD = 1e-4
AGENT_STM_BALANCE_TOLERANCE = 5
AGENT_STM_SIZE = int(1e5)
AGENT_STM_USE_MA_SMOOTHING = False
AGENT_STM_MEAN_ERROR = False
AGENT_STM_AVERAGE_WINDOW_SIZE = 10
AGENT_STM_ATTENTION_MODE = False
AGENT_DEPTH_MODE = False
AGENT_PROBABILITY_CORRECTION = True
AGENT_ARBITRAGE_ZONE_SIZE = 0.02
AGENT_ARBITRAGE_ZONE_GUARANTEE_PERCENT = 0.00
AGENT_ARBITRAGE_BASE_MARGIN = 10
AGENT_MAX_OPEN_TRADES = 20
AGENT_NUM_ACTIONS = 20
AGENT_RECOMMENDATION_PERCENT = 0.5
AGENT_DEVICE = "cpu"
AGENT_USE_SOFTMAX = False
AGENT_USE_KALMAN_FILTER = False
AGENT_KALMAN_ALPHA = 0.05
AGENT_KALMAN_BETA = 0.01
AGENT_MA_WINDOW_SIZE = 32
AGENT_USE_LASS = False
AGENT_LASS_MODEL_FS_PATH = "/Apps/RTrader/abrehamalemu-spinoza-lass-training-cnn-10-it-5-tot.zip"
AGENT_USE_SMOOTHING = not MARKET_STATE_SMOOTHING
AGENT_CRA_SIZE = 5
AGENT_CRA_DISCOUNT = 0.7
AGENT_DRMCA_WP = 100
AGENT_TOP_K_NODES = None
AGENT_DYNAMIC_K_THRESHOLD = 0.05
AGENT_DUMP_NODES = True
AGENT_DUMP_NODES_PATH = os.path.join(BASE_DIR, "temp/graph_dumps")
AGENT_DUMP_VISITED_ONLY = True
AGENT_USE_AUTO_STATE_REPOSITORY = False
AGENT_AUTO_STATE_REPOSITORY_MEMORY_SIZE = int(5e5)
AGENT_FILESYSTEM_STATE_REPOSITORY_PATH = BASE_DIR
AGENT_MIN_DISK_SPACE = 0.1
AGENT_MIN_ABS_DISK_SPACE = None
AGENT_MODEL_USE_CACHED_MODEL = True
AGENT_MODEL_USE_TRANSITION_ONLY = True
AGENT_MODEL_EXTRA_LEN = 124
AGENT_MODEL_TEMPERATURE = 1
AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON = 1e-5
with open(os.path.join(BASE_DIR, "res/bounds/10.json"), "r") as file:
	AGENT_STATE_CHANGE_DELTA_STATIC_BOUND = sorted(list(json.load(file)))
with open(os.path.join(BASE_DIR, "res/weights/05.json"), "r") as file:
	AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_WEIGHTS = sorted(list(json.load(file)))
MODEL_SAVE_EXTENSION = "zip"
TPU_OS_KEY = "COLAB_TPU_ADDR"


MC_WORKER_STEP_TIME = 1 * 60
MC_WORKERS = 8
CURRENCIES = [
	"AUD",
	"CAD",
	"CHF",
	"CZK",
	"DKK",
	"GBP",
	"HKD",
	"HUF",
	"JPY",
	"MXN",
	"NOK",
	"NZD",
	"PLN",
	"SEK",
	"SGD",
	"THB",
	"TRY",
	"USD",
	"ZAR"
]

CORE_MODEL_CONFIG = ModelConfig(
	id="core",
	url="https://www.dropbox.com/s/9nvcas994dpzq3a/model.h5?dl=0&raw=0",
	path="/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-training-cnn-1-it-42-tot.zip",
	download=False
)

DELTA_MODEL_CONFIG = ModelConfig(
	id="delta",
	url="https://www.dropbox.com/s/io0fbl7m44e6k8a/delta-bb_wrapped.h5?dl=0",
	path=os.path.join(BASE_DIR, "res/m10/combined_trained/delta_model_d.h5"),
	download=False
)

ARA_MODEL_CONFIG = ModelConfig(
	id="ara",
	url="",
	path=None,
	download=False
)

PREDICTION_MODELS = [
	CORE_MODEL_CONFIG,
	DELTA_MODEL_CONFIG,
	ARA_MODEL_CONFIG
]


MAPLOSS_FS_MODELS_PATH = "/Apps/RTrader/maploss/it-64/"


class ResourceCategories:

	TEST_RESOURCE = "test"
	RUNNER_STAT = "runner_stats"
	OANDA_ACCOUNTS = "oanda_accounts"


class RunnerStatsBranches:

	main = "main"

	ma_ews_dynamic_k_stm_it_23 = "ma_ews_dynamic_k_stm_it_23"
	it_23_0 = "it_23_0"
	it_23_1 = "it_23_1"
	it_23_2 = "it_23_2"
	it_23_3 = "it_23_3"
	ma_ews_dynamic_k_stm_it_24 = "ma_ews_dynamic_k_stm_it_24"
	ma_ews_dynamic_k_stm_it_27_mts_0_b_1 = "ma_ews_dynamic_k_stm_it_27_mts_0_b_1"
	it_27_0 = "it_27_0"  # STM = False, Step Time = 6 min
	it_27_1 = "it_27_1"
	it_27_2 = "it_27_2"
	ma_ews_dynamic_k_stm_it_29 = "ma_ews_dynamic_k_stm_it_29"
	ma_ews_dynamic_k_stm_it_29_dm_0 = "ma_ews_dynamic_k_stm_it_29_dm_0"
	it_30_1 = "it_30_1"
	ma_ews_dynamic_k_stm_it_31 = "ma_ews_dynamic_k_stm_it_31"
	it_31_2 = "it_31_2"
	ma_ews_dynamic_k_stm_it_33 = "ma_ews_dynamic_k_stm_it_33"
	it_34_1 = "it_34_1"
	it_36_1 = "it_36_1"
	it_36_2 = "it_36_2"
	it_37_1 = "it_37_1"
	it_38_1 = "it_38_1"
	it_38_2 = "it_38_2"
	it_39_1 = "it_39_1"
	it_39_2 = "it_39_2"
	it_40_2 = "it_40_2"

	it_41_2 = "it_41_2"
	it_41_6 = "it_41_6"
	it_42_2 = "it_42_2"
	it_42_4 = "it_42_4"
	it_42_5 = "it_42_5"
	it_42_6 = "it_42_6"

	it_43_2 = "it_43_2"
	it_44_2 = "it_44_2"

	it_45_6 = "it_45_6"
	it_46_6 = "it_46_6"

	it_47_6 = "it_47_6"

	it_48_6 = "it_48_6"

	it_49_6 = "it_49_6"
	it_50_6 = "it_50_6"

	it_51_6 = "it_51_6"
	it_52_6 = "it_52_6"

	it_53_6 = "it_53_6"

	it_54_6 = "it-54_6"

	it_56_6 = "it_56_6"
	it_56_7 = "it_56_7"
	it_57_6 = "it_57_6"
	it_57_8 = "it_57_8"
	it_57_9 = "it_57_9"
	it_57_10 = "it_57_10"

	it_58_6 = "it_58_6"
	it_64_6 = "it_64_6"

	it_60_6 = "it_60_6"
	it_60_7 = "it_60_7"
	it_61_6 = "it_61_6"

	it_63_6 = "it_63_6"

	all = [
		main,
		ma_ews_dynamic_k_stm_it_23,
		it_23_0,
		it_23_1,
		it_23_2,
		it_23_3,
		ma_ews_dynamic_k_stm_it_24,
		ma_ews_dynamic_k_stm_it_27_mts_0_b_1,
		it_27_0,
		it_27_1,
		it_27_2,
		ma_ews_dynamic_k_stm_it_29,
		ma_ews_dynamic_k_stm_it_29_dm_0,
		it_30_1,
		ma_ews_dynamic_k_stm_it_31,
		it_31_2,
		ma_ews_dynamic_k_stm_it_33,
		it_34_1,
		it_36_1,
		it_36_2,
		it_37_1,
		it_38_1,
		it_38_2,
		it_39_1,
		it_39_2,
		it_40_2,
		it_41_2,
		it_41_6,
		it_42_2,
		it_42_4,
		it_42_5,
		it_42_6,
		it_43_2,
		it_44_2,
		it_45_6,
		it_46_6,
		it_47_6,
		it_48_6,
		it_49_6,
		it_50_6,
		it_51_6,
		it_52_6,
		it_53_6,
		it_54_6,
		it_56_6,
		it_56_7,
		it_57_6,
		it_57_8,
		it_57_9,
		it_57_10,
		it_58_6,
		it_64_6,
		it_60_6,
		it_60_7,
		it_61_6,
		it_63_6
	]

	default = it_64_6


class RunnerStatsLossesBranches:

	main = "main"
	it_23 = "it_23"
	it_23_sw_0 = "it_23_sw_0"
	it_23_sw_1 = "it_23_sw_1"
	it_23_sw_2 = "it_23_sw_2"
	it_23_sw_3 = "it_23_sw_3"
	it_23_sw_4 = "it_23_sw_4"
	it_23_sw_5 = "it_23_sw_5"
	it_23_sw_6 = "it_23_sw_6"
	it_23_sw_7 = "it_23_sw_7"
	it_23_sw_8 = "it_23_sw_8"
	it_23_sw_9 = "it_23_sw_9"
	it_23_sw_10 = "it_23_sw_10"
	it_23_sw_11 = "it_23_sw_11"
	it_23_sw_12 = "it_23_sw_12"
	it_24 = "it_24"
	it_27 = "it_27"
	it_27_sw_11 = "it_27_sw_11"
	it_27_sw_12 = "it_27_sw_12"
	it_29 = "it_29"
	it_30 = "it_30"
	it_31 = "it_31"
	it_33 = "it_33"
	it_34 = "it_34"
	it_36 = "it_36"
	it_37 = "it_37"
	it_38 = "it_38"
	it_39 = "it_39"
	it_40 = "it_40"
	it_41 = "it_41"
	it_41_h_0 = "it_41_h_0"
	it_41_h_1 = "it_41_h_1"
	it_41_0 = "it_41_0"
	it_41_1 = "it_41_1"
	it_42 = "it_42"
	it_43_1 = "it_43_1"
	it_44_1 = "it_44_1"
	it_45_0 = "it_45_0"
	it_47_0 = "it_47_0"
	it_49_0 = "it_49_0"
	it_51_0 = "it_51_0",
	it_54_0 = "it_54_0"
	it_56_0 = "it_56_0"
	it_58_0 = "it_58_0"
	it_60_0 = "it_60_0"

	all = [
		main,
		it_23,
		it_23_sw_0,
		it_23_sw_1,
		it_23_sw_2,
		it_23_sw_3,
		it_23_sw_4,
		it_23_sw_5,
		it_23_sw_6,
		it_23_sw_7,
		it_23_sw_8,
		it_23_sw_9,
		it_23_sw_10,
		it_23_sw_11,
		it_23_sw_12,
		it_24,
		it_27,
		it_27_sw_11,
		it_29,
		it_30,
		it_31,
		it_33,
		it_34,
		it_36,
		it_37,
		it_38,
		it_39,
		it_40,
		it_41,
		it_41_h_0,
		it_41_h_1,
		it_41_0,
		it_41_1,
		it_42,
		it_43_1,
		it_44_1,
		it_45_0,
		it_47_0,
		it_49_0,
		it_51_0,
		it_54_0,
		it_56_0,
		it_58_0,
		it_60_0
	]

	default = it_58_0
