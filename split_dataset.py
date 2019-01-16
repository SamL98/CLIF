import os
from os.path import join, isdir
from numpy.random import choice

img_dir = 'E:/lerner/clif/img'
gt_dir = 'E:/lerner/clif/gt'

num_train = 900
num_val = 96

val_idxes = choice(range(num_train+num_val), num_val, replace=False)
n_train, n_val = 0, 0

fnames = [fname[:fname.index('.')] for fname in os.listdir(img_dir) if (not isdir(join(img_dir, fname)))]
for i, fname in enumerate(fnames):
	imset = 'train'
	if i in val_idxes:
		imset = 'val'
		n_val += 1
		idx = n_val
	else:
		n_train += 1
		idx = n_train
	
	os.rename(join(img_dir, fname+'.png'), join(img_dir, imset, '%s_img_%06d.png' % (imset, idx)))
	os.rename(join(gt_dir, fname+'.mat'), join(gt_dir, imset, '%s_pixeltruth_%06d.mat' % (imset, idx)))
	