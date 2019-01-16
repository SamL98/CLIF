from keras.callbacks import Callback
from os.path import isfile

class Logger(Callback):
	def __init__(self, fname):
		self.fname = fname
		
		if not isfile(fname):
			with open(fname, 'w') as f:
				f.write('epoch,loss,val_loss\n')
			
	def on_epoch_end(self, epoch, logs={}):
		loss = logs['loss']
		val_loss = logs['val_loss']
		with open(self.fname, 'a') as f:
			f.write('%d,%f,%f\n' % (epoch, loss, val_loss))