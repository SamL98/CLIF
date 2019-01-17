from load_data import _DS_INFO, load_image

'''
n = 1
mean = 0
stdev = 0

for i in range(1, _DS_INFO['num_train']+1):
	img = load_image('train', i)
	mean += (img.mean() - mean)/n
	stdev += (img.std() - stdev)/n
	n += 1
	
for i in range(1, _DS_INFO['num_val']+1):
	img = load_image('val', i)
	mean += (img.mean() - mean)/n
	stdev += (img.std() - stdev)/n
	n += 1
'''

imgs = []
for i in range(1, _DS_INFO['num_train']+1):
	imgs.append(load_image('train', i))
for i in range(1, _DS_INFO['num_val']+1):
	imgs.append(load_image('val', i))
	
import numpy as np
imgs = np.array(imgs)
mean, stdev = imgs.mean(), imgs.std()
	
print('Mean: %f, Stdev: %f' % (mean, stdev))