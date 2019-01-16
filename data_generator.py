from keras.preprocessing.image import ImageDataGenerator

def image_data_generator():
	idg = ImageDataGenerator(brightness_range=(0.85, 1.15),
							horizontal_flip=True)
	return idg 
	
if __name__ == '__main__':
	import numpy as np
	from load_data import load_dataset
	
	X_val, y_val = load_dataset('val')
	val_dg = data_generator()
	
	val_iter = val_dg.flow(X_val, y_val)
	
	done = False
	gs, mask = None, None
	
	while not done:
		x_batch, y_batch = next(val_iter)
		for x, y in zip(x_batch, y_batch):
			if np.sum(y[...,1])>0:
				gs = x
				mask = y[...,1]
				
				done = True
				break
	
	import matplotlib.pyplot as plt
	_, ax = plt.subplots(1, 5)
	for i in range(4):
		ax[i].imshow(x[...,i], 'gray')
	ax[4].imshow(mask, 'jet')
	plt.show()