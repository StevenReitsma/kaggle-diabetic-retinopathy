class OvRConverter(object):
	"""
	`number` indicates the split. 0 means 0 <> 1-4, 1 means 0-1 <> 2-4, etc.
	"""
	def __init__(self, number):
		self.number = number

	def transform(self, labels):
		labels[labels <= self.number] = 0
		labels[labels > self.number] = 1

		return labels