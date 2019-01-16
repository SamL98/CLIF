import numpy as np
from scipy.ndimage import gaussian_filter

def get_grads(img, sigma=2):
	gx = gaussian_filter(img, sigma=sigma, order=(0, 1))[...,np.newaxis]/255.
	gy = gaussian_filter(img, sigma=sigma, order=(1, 0))[...,np.newaxis]/255.
	
	c = 1/np.sqrt(2.)
	s = 1/np.sqrt(2.)
	
	gpo4 = c*gx + s*gy
	gnpo4 = c*gx - s*gy
	
	return np.concatenate([gx, gy, gpo4, gnpo4], axis=2)
	
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from skimage.io import imread
	
	idx = np.random.choice(range(96))
	img = imread('E:lerner/clif/img/val/val_img_%06d.png' % (idx+1))
	grads = get_grads(img[...,np.newaxis])
	
	_, ax = plt.subplots(1, 5)
	ax[0].imshow(img, 'gray')
	for i in range(1, 5):
		ax[i].imshow(grads[...,(i-1)], 'gray')
	plt.show()