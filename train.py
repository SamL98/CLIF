from load_data import gen_imset, _DS_INFO, _NUM_CHANNELS
from model import create_model

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
model = None

def weighted_xentropy(y_true, y_pred):
	bs = BATCH_SIZE
	
	# Steps are as follows:
	#	1. Multiply the second channel of the ground truth by 5 to weigh building more heavily
	#		a. Now `weighted_gt` has 1's in the first channel where there is no building
	#			and 5's in the second channel where there is building
	#
	#	2. Take the log of the predicted softmax probabilities
	#
	#	3. Multiply `weighted_gt` by the log probs
	#		a. Now `masked_log_pred` has:
	#			i. In the first channel, log(p_nb) where nb=True and 0 elsewhere
	#			ii. In the second channel, 5*log(p_b) where b=True and 0 elsewhere
	#	
	#	4. Take the maximum value along the channel axis so the log prob of the correct class is there
	#
	#	5. Return the mean
	
	nb_true_slice = K.slice(y_true, [0, 0, 0, 0], [bs, 512, 512, 1])
	b_true_slice = K.slice(y_true, [0, 0, 0, 1], [bs, 512, 512, 1]) * 3 # building weight
	weighted_gt = K.concatenate([nb_true_slice, b_true_slice])
	
	log_pred = -K.log(K.maximum(y_pred, 1e-5))
	masked_log_pred = log_pred * weighted_gt
	correct_class_log_pred = K.max(masked_log_pred, axis=-1)
	
	return K.mean(correct_class_log_pred)
	
	log_probs = -K.log(K.maximum(y_pred, 1e-5))
	nb_pred_slice = K.slice(log_probs, [0, 0, 0, 0], [bs, 512, 512, 1])
	b_pred_slice = K.slice(log_probs, [0, 0, 0, 1], [bs, 512, 512, 1])
	
	#num_pix = K.sum(nb_true_slice) + K.sum(b_true_slice)
	weighted_nb_loss = (nb_pred_slice * nb_true_slice) / K.sum(nb_true_slice)# / num_pix
	weighted_b_loss = (b_pred_slice * b_true_slice) / K.sum(b_true_slice)# / num_pix
	
	return weighted_nb_loss + weighted_b_loss

	
def dice_loss(y_true, y_pred):
	gt = K.argmax(y_true, axis=-1)
	pred = K.argmax(y_pred, axis=-1)
	intersection = K.sum(gt*pred)
	
	smooth = 1
	dice_coef = (2*intersection + smooth) / (K.sum(gt) + K.sum(pred) + smooth)
	return 1 - dice_coef
	
def iou_loss(y_true, y_pred):
	gt = K.cast(K.argmax(y_true, axis=-1), dtype=tf.float32)
	pred = K.cast(K.argmax(y_pred, axis=-1), dtype=tf.float32)
	intersection = K.sum(gt*pred)
	union = K.sum(gt)+K.sum(pred)-intersection
	return intersection/union
	

#optimizer = SGD(momentum=0.99)
optimizer = Adam()
#loss = 'categorical_crossentropy'
loss = weighted_xentropy
#loss = iou_loss
	
if not isdir(join('models', name)):
	mkdir(join('models', name))
	
	model = create_model(_DS_INFO['num_classes'], _NUM_CHANNELS)
	model.compile(optimizer, loss=loss)
elif isfile(join('models', name, 'model.h5')):
	with open(join('models', name, 'model.json')) as f:
		json = f.read()
	
	model = model_from_json(json)
	model.load_weights(join('models', name, 'model.h5'))
	model.compile(optimizer, loss=loss)

checkpoint = ModelCheckpoint(join('models', name, 'model.h5'), 
							monitor='val_loss', 
							mode='min',
							verbose=1,
							save_best_only=True)
							
log = Logger(join('models', name, 'metrics.csv'))

with open(join('models', name, 'model.json'), 'w') as f:
	f.write(model.to_json())

model.fit_generator(train_dg,
					steps_per_epoch=_DS_INFO['num_train']//BATCH_SIZE,
					epochs=100,
					callbacks=[early_stop, checkpoint, log],
					validation_data=val_dg,
					validation_steps=_DS_INFO['num_val']//BATCH_SIZE)