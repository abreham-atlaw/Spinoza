
class CurrencyNotFoundException(Exception):

	def __init__(self, currency):
		self.currency = currency

	def __str__(self):
		return "Currency not found: " + self.currency


class InsufficientFundsException(Exception):

	def __init__(self, available_margin: float, requested_margin: float):
		super().__init__(f"Available margin: {available_margin} is less than requested margin: {requested_margin}")
