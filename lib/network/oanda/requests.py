from typing import *

import json
from datetime import datetime

from lib.network.rest_interface import Request
from .data.models import AccountSummary, Trade, Order, CreateOrderResponse, CloseTradeResponse, Price, CandleStick


class AccountSummaryRequest(Request):

	def __init__(self):
		super().__init__("accounts/{{account_id}}/summary/", output_class=AccountSummary)

	def _filter_response(self, response: Dict) -> Dict:
		return response["account"]


class GetOpenTradesRequest(Request):

	def __init__(self):
		super().__init__("accounts/{{account_id}}/openTrades/", output_class=List[Trade])

	def _filter_response(self, response: Dict) -> Dict:
		return response["trades"]


class CreateOrderRequest(Request):

	def __init__(self, order: Order):
		super().__init__(
			"accounts/{{account_id}}/orders/",
			method=Request.Method.POST, 
			post_data=order, 
			output_class=CreateOrderResponse,
			headers={"Content-Type": "application/json"}
		)
	
	def get_post_data(self) -> Dict:
		return json.dumps({
			"order": super().get_post_data()
		})

	def _filter_response(self, response):
		if "orderFillTransaction" in response:
			return response["orderFillTransaction"]
		return response["orderCancelTransaction"]


class GetInstrumentsRequest(Request):

	def __init__(self):
		super().__init__("accounts/{{account_id}}/instruments/", output_class=List[Tuple[str, str]])
	
	def _filter_response(self, response):
		from . import Trader
		return [Trader.split_instrument(instrument["name"]) for instrument in response["instruments"]]
	

class CloseTradeRequest(Request):

	def __init__(self, id_):
		super().__init__(
			"accounts/{{account_id}}/trades/{trade_id}/close", 
			method=Request.Method.PUT,
			url_params={"trade_id": id_},
			output_class=CloseTradeResponse
		)
	
	def _filter_response(self, response):
		if "orderFillTransaction" in response:
			return response["orderFillTransaction"]
		return response["orderCancelTransaction"]


class GetPriceRequest(Request):

	def __init__(self, instrument):
		super().__init__(
			"accounts/{{account_id}}/pricing/",
			get_params={"instruments": [instrument]},
			output_class=Price
		)
	
	def _filter_response(self, response):
		return response["prices"][0]


class GetCandleSticksRequest(Request):

	def __init__(self, instrument: Tuple[str, str], from_: datetime = None, to: datetime = None, granularity: str = None, count: int = None):
		from . import Trader
		get_params = {
			"granularity": granularity,
			"count": count,
			"Accept-Datetime-Format": "UNIX",
		}
		if to is not None:
			get_params["to"] = to.timestamp()
		if from_ is not None:
			get_params["from"] = from_.timestamp()
		super().__init__(
			"accounts/{{account_id}}/instruments/{instrument}/candles/",
			url_params={"instrument": Trader.format_instrument(instrument)},
			get_params=get_params,
			output_class=List[CandleStick]
		)

	def _filter_response(self, response):
		return response["candles"]
