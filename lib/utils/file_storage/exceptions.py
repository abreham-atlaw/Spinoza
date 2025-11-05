
class FileNotFoundException(Exception):
	def __init__(self, path: str):
		super().__init__(f"File not found at path: {path}")


class FileSystemException(Exception):
	pass

