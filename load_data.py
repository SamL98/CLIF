import numpy as np
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from hdf5storage import loadmat
import json
from os.path import join

from data_generator import image_data_generator
from gradients import get_grads

_DS_DIR = 'E:/LERNER/clif'
with open(join(_DS_DIR, 'dataset_info.json')) as f:
	_DS_INFO = json.loads(f.read())
	
_DS_INFO['imsize'] = 512
_DS_INFO['num_classes'] = 2
_DS_INFO['target_class'] = 3

_IMG_ROOT_DIR = join(_DS_DIR, 'img')
_GT_ROOT_DIR = join(_DS_DIR, 'gt')
_IMG_FMT = '%s_img_%06d.png'
_GT_FMT = '%s_pixeltruth_%06d.mat'
_GT_KEY = 'ground_truth'

_PIX_MEAN = 34.417
_NUM_CHANNELS = 1

def load_image(imset, idx):
	return imread(join(_IMG_ROOT_DIR, imset, _IMG_FMT % (imset, idx)))[:-1, :-1]
	
def load_gt(imset, idx):
	return loadmat(join(_GT_ROOT_DIR, imset, _GT_FMT % (imset, idx)))[_GT_KEY][:-1, :-1]
	
def prepro_image(img):
	x = np.zeros((_DS_INFO['imsize'], _DS_INFO['imsize'], _NUM_CHANNELS), dtype=np.float32)
	x[...,0] = (img[:]-_PIX_MEAN)/255.
	return x
	#grads = get_grads(img)
	#x[...,1:] = grads[...,:2]
	#return x
	
def prepro_gt(gt):
	new_gt = np.zeros((gt.shape[0], gt.shape[1], 2), dtype=np.uint8)
	new_gt[gt != _DS_INFO['target_class'], 0] = 1
	new_gt[gt == _DS_INFO['target_class'], 1] = 1
	return new_gt
	
def gen_imset(imset, batch_size=16, augment=True):
	imset = imset.lower()
	assert (imset == 'train' or imset == 'val')
	
	num_img = _DS_INFO['num_%s'%imset]
	img_dir = join(_IMG_ROOT_DIR, imset)
	gt_dir = join(_GT_ROOT_DIR, imset)
	
	idg = image_data_generator()
	dim = _DS_INFO['imsize']
	nc = _DS_INFO['num_classes']
	
	while True:
		idxs = np.random.choice(range(1, num_img+1), num_img, replace=False)
		
		for i in range(num_img//batch_size):
			if (i+1)*batch_size > num_img:
				all_idxs = set(list(range(1, num+img+1)))
				all_idxs.difference(idxs[i*batch_size:])
				new_idxs = np.random.choice(all_idxs, len(all_idxs), replace=False)
				idxs = np.concatenate([idxs, new_idxs])
				
			X_batch = np.zeros((batch_size, dim, dim, _NUM_CHANNELS), dtype=np.float32)
			y_batch = np.zeros((batch_size, dim, dim, nc), dtype=np.float32)
			
			for j in range(batch_size):
				idx = idxs[i*batch_size+j]
				x = imread(join(img_dir, _IMG_FMT % (imset, idx)))[:dim, :dim]
				y = loadmat(join(gt_dir, _GT_FMT % (imset, idx)))[_GT_KEY][:dim, :dim]
				
				if augment:
					x = idg.random_transform(x[...,np.newaxis])[...,0]
				X_batch[j] = prepro_image(x)
			
				y_batch[j] = prepro_gt(y)
				
			yield X_batch, y_batch
			
	