import unirest
import time
import socket

class Plotta(object):
	"""
	Communicates progress to plotta.

	param `hostname`: Hostname to push to.
	param `port`: Port to push to.
	"""
	def __init__(self, hostname, port = 80):
		self.hostname = hostname
		self.port = port

	def job_start(self, job_id, name, node):
		payload = {'id': job_id, 'name': name, 'node': node}
		url = "http://{0}:{1}/api/job/new".format(self.hostname, self.port)

		return self._do_sync_request(url, payload)

	def job_finished(self, job_id):
		payload = {'id': job_id}
		url = "http://{0}:{1}/api/job/stop".format(self.hostname, self.port)

		self._do_async_request(url, payload)

	def new_stream(self, stream_id, job_id, title, x_name, y_name):
		payload = {'id': stream_id, 'job_id': job_id, 'title': title, 'xName': x_name, 'yName': y_name}
		url = "http://{0}:{1}/api/stream/new".format(self.hostname, self.port)

		self._do_async_request(url, payload)

	def append(self, stream_id, job_id, x, y):
		payload = {'id': stream_id, 'job_id': job_id, 'x': x, 'y': y}
		url = "http://{0}:{1}/api/stream/append".format(self.hostname, self.port)

		self._do_async_request(url, payload)

	def _do_sync_request(self, url, payload):
		return unirest.post(url, headers = {"Accept": "application/json"}, params = payload)

	def _do_async_request(self, url, payload):
		def empty_callback(response):
			pass

		unirest.post(url, headers = {"Accept": "application/json"}, params = payload, callback = empty_callback)


class PlottaDiabetic(Plotta):
	def __init__(self, job_name, hostname, port = 80):
		super(PlottaDiabetic, self).__init__(hostname, port)

		self.job_name = job_name
		self.job_id = int(round(time.time()))
		self.node_name = socket.getfqdn()
		self.disabled = False

	def start(self):
		r = self.job_start(self.job_id, self.job_name, self.node_name)

		if r.code in [404, 502, 503, 504]:
			# Server offline, disable plotta
			print "Plotta server offline. Disabling plotta for this job."
			self.disabled = True
		elif r.code in [400, 405, 406, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 501]:
			raise RuntimeError("Plotta user error ({0}). Message: {1}".format(r.code, r.body))
		elif r.code in [500]:
			raise RuntimeError("Plotta server error ({0}). Message: ".format(r.code, r.body))

		self.new_stream("kappa", self.job_id, "Kappa", "Epochs", "Kappa")
		self.new_stream("train_mse", self.job_id, "Train MSE", "Epochs", "Train MSE")
		self.new_stream("valid_mse", self.job_id, "Valid MSE", "Epochs", "Valid MSE")
		self.new_stream("duration", self.job_id, "Duration per epoch", "Epochs", "Computation time (seconds)")
		self.new_stream("overfitting", self.job_id, "Overfitting", "Epochs", "Train MSE / Valid MSE")

	def stop(self):
		if self.disabled:
			return

		self.job_finished(self.job_id)

	def add(self, epoch, kappa, train_mse, valid_mse, duration, overfit = None):
		if self.disabled:
			return

		if overfit is None:
			overfit = train_mse / valid_mse

		self.append("kappa", self.job_id, epoch, kappa)
		self.append("train_mse", self.job_id, epoch, train_mse)
		self.append("valid_mse", self.job_id, epoch, valid_mse)
		self.append("duration", self.job_id, epoch, duration)
		self.append("overfitting", self.job_id, epoch, overfit)


class PlottaStart():
	"""
	Nolearn hook wrapper.
	"""
	def __init__(self, plotta):
		self.plotta = plotta

	def __call__(self, nn, train_history):
		self.plotta.start()


class PlottaUpdate():
	"""
	Nolearn hook wrapper.
	"""
	def __init__(self, plotta):
		self.plotta = plotta

	def __call__(self, nn, train_history):
		self.plotta.add(train_history[-1]['epoch'], train_history[-1]['kappa'], train_history[-1]['train_loss'], train_history[-1]['valid_loss'], train_history[-1]['dur'])


class PlottaStop():
	"""
	Nolearn hook wrapper.
	"""
	def __init__(self, plotta):
		self.plotta = plotta

	def __call__(self, nn, train_history):
		self.plotta.stop()
