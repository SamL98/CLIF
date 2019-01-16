from keras.models import model_from_json
from os.path import join

from load_data import prepro_gt, load_image, load_gt, _DS_INFO, prepro_image
from gradients import get_grads

import numpy as np
from numpy import random
from skimage.io import imread
from hdf5storage import loadmat
import matplotlib.pyplot as plt

import sys
name = sys.argv[1]
model_dir = join('models', name)

with open(join(model_dir, 'model.json')) as f:
	json = f.read()
	
model = model_from_json(json)
model.load_weights(join(model_dir, 'model.h5'))

def display_random_prediction(imset, idx=None, data=None):
	if idx is None:
		idx = random.choice(range(1, _DS_INFO['num_%s'%imset]+1), 1)
	print(idx)
	img = load_image(imset, idx)
	gt = load_gt(imset, idx)
	
	if data is None:
		X = np.expand_dims(prepro_image(img), axis=0)
		mask = prepro_gt(gt)[...,1]
		pred = np.argmax(model.predict(X)[0], axis=2)
	else:
		X, mask, pred = data
	
	fig, ax = plt.subplots(1, 3)
	ax[0].imshow(img, 'gray')
	ax[0].set_title('Original')
	
	ax[1].imshow(img, 'gray')
	ax[1].imshow(pred, 'OrRd', alpha=0.3)
	ax[1].set_title('Predicted')
	
	ax[2].imshow(img, 'gray')
	ax[2].imshow(mask, 'OrRd', alpha=0.3)
	ax[2].set_title('Ground Truth')
	
	fig.savefig('%s_%s_%d.png' % (name, imset, idx))
	plt.show()
	
def display_prediction_w_building(imset):
	for i in range(1, _DS_INFO['num_%s'%imset]+1):
		img = load_image(imset, i)
		gt = load_gt(imset, i)
		
		X = np.expand_dims(prepro_image(img), axis=0)
		mask = prepro_gt(gt)[...,1]
		pred = np.argmax(model.predict(X)[0], axis=2)
		
		if (pred==1).sum() > 0:
			display_random_prediction(imset, idx=i, data=(X, mask, pred))
			return
			

idx = None
if len(sys.argv) > 2:
	idx = int(sys.argv[2])
	
display_random_prediction('val', idx=idx)
#display_prediction_w_building('val')