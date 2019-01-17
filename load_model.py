from os.path import isdir, isfile, join
from os import mkdir
from load_data import _DS_INFO, _NUM_CHANNELS

def load_model(model_dir):
	model_description_file = join(model_dir, 'model.json')
	model_weights_file = join(model_dir, 'model.h5')
	
	assert isfile(model_description_file)
	assert isfile(model_weights_file)

	with open(model_description_file) as f:
		json = f.read()
	
	from keras.models import model_from_json
	model = model_from_json(json)
	model.load_weights(model_weights_file)
	return model

def create_model(model_dir):
	mkdir(model_dir)
	
	from model import create_model
	model = create_model(_DS_INFO['num_classes'], _NUM_CHANNELS)
	return model

def load_or_create_model(name):
	model_dir = join('models', name)
	
	if not isdir(model_dir):
		return create_model(model_dir)
	else:
		return load_model(model_dir)