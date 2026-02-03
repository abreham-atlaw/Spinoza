import socketio


class SwarmSocket:

	def __init__(self, *args, **kwargs):
		self.__args = args
		self.__kwargs = kwargs
		self.__sio = self.__init_sio()

	def __init_sio(self) -> socketio.Client:
		return socketio.Client(*self.__args, **self.__kwargs)

	def __getattr__(self, name):
		return getattr(self.__sio, name)

	def reset(self):
		self.__sio.disconnect()
		self.__sio = socketio.Client(*self.__args, **self.__kwargs)