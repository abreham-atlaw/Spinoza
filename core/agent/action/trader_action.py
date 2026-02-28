from dataclasses import dataclass

from .action import Action


@dataclass
class TraderAction(Action):
	class Action:
		BUY = 1
		CLOSE = 2
		SELL = 0

	base_currency: str
	quote_currency: str
	action: int
	margin_used: float = None
	units: int = None
	stop_loss: float = None
	take_profit: float = None

	def __eq__(self, other):
		if not isinstance(other, TraderAction):
			return False
		return \
			self.base_currency == other.base_currency and \
			self.quote_currency == other.quote_currency and \
			self.action == other.action and \
			self.stop_loss == other.stop_loss and \
			self.take_profit == other.take_profit and \
			(
				self.margin_used == other.margin_used or
				self.units == other.units
			)

	def __deepcopy__(self, memo=None):
		return TraderAction(
			base_currency=self.base_currency,
			quote_currency=self.quote_currency,
			action=self.action,
			margin_used=self.margin_used,
			units=self.units,
			stop_loss=self.stop_loss,
			take_profit=self.take_profit
		)

	def __hash__(self):
		return hash((self.base_currency, self.quote_currency, self.action, self.margin_used, self.units, self.stop_loss, self.take_profit))
