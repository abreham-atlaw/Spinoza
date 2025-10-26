import unittest

from core.utils.research.utils.training_plotter import TrainingPlotter


class TrainingPlotterTest(unittest.TestCase):

	def setUp(self):
		self.plotter = TrainingPlotter(threshold=20)

	def test_plot(self):
		self.plotter.plot(
			"abrehamalemu-spinoza-lass-training-cnn-18-it-5-tot"
		)

	def test_plot_multiple(self):
		self.plotter.plot_multiple([
			f"abrehamalemu-rtrader-training-exp-0-cnn-{i}-cum-0-it-27-sw12-tot"
			for i in [
				66, 67, 68, 69, 71
			]
		] + [
			f"abrehamalemu-rtrader-training-exp-0-cnn-{i}-cum-0-it-37-tot"
			for i in [
				1
			]
		] + [
			f"abrehamalemu-rtrader-training-exp-0-cnn-{i}-cum-0-it-38-tot"
			for i in [
				0, 1
			]
		])
