from load_data import gen_imset, _DS_INFO, _NUM_CHANNELS
from load_model import load_or_create_model
from loss import get_model_loss

import tensorflow as tf
#from tensorflow.nn import softmax_cross_entropy_with_logits as xentropy
import keras.backend as K

from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from logger import Logger

from keras.optimizers import Adam, SGD

from os.path import join, isdir, isfile
from os import mkdir

import argparse
parser = argparse.ArgumentParser(description='Train a U-Net on the CLIF2007 dataset')
parser.add_argument('name', type=str, help='The name for the current model being trained (required).')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='The minibatch size for training')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
train_dg = gen_imset('train', batch_size=BATCH_SIZE)
val_dg = gen_imset('val', batch_size=BATCH_SIZE)

early_stop = EarlyStopping(patience=15, verbose=1)

name = args.name
model_dir = join('models', name)

#optimizer = SGD(momentum=0.99)
optimizer = Adam()

model = load_or_create_model(name)
model.compile(optimizer, loss=get_model_loss(BATCH_SIZE))

checkpoint = ModelCheckpoint(join(model_dir, 'model.h5'), 
							monitor='val_loss', 
							mode='min',
							verbose=1,
							save_best_only=True)
							
log = Logger(join(model_dir, 'metrics.csv'))

with open(join(model_dir, 'model.json'), 'w') as f:
	f.write(model.to_json())

model.fit_generator(train_dg,
					steps_per_epoch=_DS_INFO['num_train']//BATCH_SIZE,
					epochs=100,
					callbacks=[early_stop, checkpoint, log],
					validation_data=val_dg,
					validation_steps=_DS_INFO['num_val']//BATCH_SIZE)