from load_data import _DS_INFO, load_gt, prepro_gt
import numpy as np

mean_building_prob = 0
n = 1
dim = _DS_INFO['imsize']

for i in range(1, _DS_INFO['num_train']+1):
	gt = prepro_gt(load_gt('train', i))
	building_prob = float((np.argmax(gt, axis=2)==1).sum())/(dim*dim)
	mean_building_prob += (building_prob - mean_building_prob)/n
	n += 1
	
print(mean_building_prob)