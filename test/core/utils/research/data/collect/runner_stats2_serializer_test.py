import unittest

import datetime

import numpy as np

from core.utils.research.data.collect.runner_stats2 import RunnerStats2
from core.utils.research.data.collect.runner_stats2_serializer import RunnerStats2Serializer


class RunnerStats2SerializerTest(unittest.TestCase):

	def setUp(self):
		self.serializer = RunnerStats2Serializer()

	def test_v1(self):
		data = {
			'id': 'abrehamalemu-spinoza-training-cnn-0-it-75-tot.0.-(T=0.1)',
			'model_name': 'abrehamalemu-spinoza-training-cnn-0-it-75-tot.0.zip',
			'session_timestamps': [
				datetime.datetime(2025, 11, 28, 12, 1, 58, 592000),
				datetime.datetime(2025, 11, 28, 18, 14, 31, 271000),
				datetime.datetime(2025, 11, 29, 14, 43, 43, 536000),
				datetime.datetime(2025, 11, 29, 21, 14, 19, 872000),
				datetime.datetime(2025, 11, 30, 4, 7, 30, 799000),
				datetime.datetime(2025, 11, 30, 10, 52, 28, 783000),
				datetime.datetime(2025, 11, 30, 17, 54, 27, 175000)
			],
			'simulated_timestamps': [
				'2024-02-28 13:07:00+00:00',
				'2024-03-28 21:30:00+00:00',
				'2024-01-21 14:08:00+00:00',
				'2023-11-21 13:44:00+00:00',
				'2024-02-07 04:05:00+00:00',
				'2024-02-16 14:53:00+00:00',
				'2024-03-27 01:54:00+00:00'
			],
			'profits': [
				-43.69176340938809,
				-1.8026410881561787,
				35.39967550876378,
				-33.826323992412924,
				-77.14529674931444,
				83.37493207489157,
				7.735962749501226
			],
			'duration': 151261.914868,
			'model_losses_map': {'it_74_0': [
				4.318386554718018,
				8.437211036682129,
				0.8564754724502563,
				0.04428574815392494,
				np.nan,
				np.nan,
				46.720741271972656,
				5.404362201690674,
				32.31105041503906,
				96.76014709472656,
				44.295928955078125,
				45.88665771484375,
				279.9362487792969,
				5006.3125,
				3826.444091796875,
				7.783874988555908
			]},
			'temperature': 0.1,
			'session_model_losses': [
				25.375,
				29.906780242919922,
				30.491666793823242,
				31.350000381469727,
				22.65833282470703,
				24.299999237060547,
				20.16666603088379
			]
		}

		stat = self.serializer.deserialize(data)
		self.assertIsInstance(stat, RunnerStats2)
		self.assertEqual(stat.id, data["id"])
		self.assertEqual(stat.model_name, data["model_name"])
		self.assertEqual(stat.session_timestamps, data["session_timestamps"])
		self.assertEqual(stat.simulated_timestamps, data["simulated_timestamps"])
		self.assertEqual(stat.profits, data["profits"])
		self.assertEqual(stat.duration, data["duration"])
		self.assertEqual(stat.model_losses_map, data["model_losses_map"])
		self.assertEqual(stat.temperature, data["temperature"])
		self.assertEqual(stat.session_model_losses, data["session_model_losses"])
		self.assertEqual(stat.sessions[0].session_timestamp, data["session_timestamps"][0])
		self.assertEqual(stat.sessions[0].simulated_timestamp, data["simulated_timestamps"][0])
		self.assertEqual(stat.sessions[0].profit, data["profits"][0])
		self.assertEqual(stat.sessions[0].model_loss, data["session_model_losses"][0])


		print(stat)

		stat.add_session_timestamp(datetime.datetime.now())
		print(stat.get_active_session())
		stat.add_profit(1.0)
		print(stat.get_active_session())
		stat.add_simulated_timestamp("test timestamp")
		print(stat.get_active_session())
		stat.add_session_model_loss(1.0)
		print(stat.sessions[-1])

		new_data = self.serializer.serialize(stat)
		self.assertIsInstance(new_data, dict)
		self.assertIsInstance(new_data["sessions"], list)
		self.assertIsInstance(new_data["sessions"][0], dict)
		print(new_data)
		print(new_data["sessions"])