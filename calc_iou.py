from load_data import gen_imset, _DS_INFO
import numpy as np
from os.path import join
from keras.models import model_from_json

def calc_iou(y_true, y_pred):
	gt = np.argmax(y_true, axis=-1)
	pred = np.argmax(y_pred, axis=-1)
	return float((gt & pred).sum()) / max((gt | pred).sum(), 1e-6)
	
def calc_batch_iou(y_batch, pred_batch):
	batch_iou = 0
	n = 1
	
	for i in range(len(y_batch)):
		iou = calc_iou(y_batch[i], pred_batch[i])
		batch_iou += (iou-batch_iou)/n
		n += 1
		
	return batch_iou

if __name__ == '__main__':
	import sys
	name = sys.argv[1]
	model_dir = join('models', name)

	with open(join(model_dir, 'model.json')) as f:
		json = f.read()
	
	model = model_from_json(json)
	model.load_weights(join(model_dir, 'model.h5'))

	imset = 'val'
	num_img = _DS_INFO['num_%s' % imset]
	batch_size = 4
	
	assert num_img % batch_size == 0
	
	dg = gen_imset(imset, batch_size=batch_size, augment=False)
	num_batch = num_img//batch_size

	mean_iou = 0
	n = 1

	for _ in range(num_batch):
		X_batch, y_batch = next(dg)
		pred_batch = model.predict(X_batch)
		batch_iou = calc_batch_iou(y_batch, pred_batch)
		mean_iou += (batch_iou - mean_iou)/n
		n += 1
		
	print('mIoU on %s set is: %f' % (imset, mean_iou))