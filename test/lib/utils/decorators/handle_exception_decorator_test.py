import unittest

from lib.utils.decorators import handle_exception


class DummyException(Exception):
	pass

class HandleExceptionDecoratorTest(unittest.TestCase):


	@handle_exception(exception_cls=(DummyException,))
	def dummy_function(self):
		raise DummyException()

	def test_handle_exception(self):
		exception_handled = False
		try:
			x = self.dummy_function()
			exception_handled = True
		except Exception as e:
			exception_handled = False

		self.assertTrue(exception_handled)
