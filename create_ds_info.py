import json
from os.path import join
import os

ds_dir = 'E:/lerner/clif'
num_train = len(os.listdir(join(ds_dir, 'img', 'train')))
num_val = len(os.listdir(join(ds_dir, 'img', 'val')))

ds_info = {
	'num_train': num_train,
	'num_val': num_val
}

with open(join(ds_dir, 'dataset_info.json'), 'w') as f:
	f.write(json.dumps(ds_info))