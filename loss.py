import keras.backend as K

def batch_weighted_xentropy(batch_size):
	def weighted_xentropy(y_true, y_pred):
		nb_true_slice = K.slice(y_true, [0, 0, 0, 0], [batch_size, 512, 512, 1])
		b_true_slice = K.slice(y_true, [0, 0, 0, 1], [batch_size, 512, 512, 1]) * 4 # building weight
		weighted_gt = K.concatenate([nb_true_slice, b_true_slice])
	
		log_pred = -K.log(K.maximum(y_pred, 1e-5))
		masked_log_pred = log_pred * weighted_gt
		correct_class_log_pred = K.max(masked_log_pred, axis=-1)
	
		return K.mean(correct_class_log_pred)
	
		#log_probs = -K.log(K.maximum(y_pred, 1e-5))
		#nb_pred_slice = K.slice(log_probs, [0, 0, 0, 0], [bs, 512, 512, 1])
		#b_pred_slice = K.slice(log_probs, [0, 0, 0, 1], [bs, 512, 512, 1])
	
		#weighted_nb_loss = (nb_pred_slice * nb_true_slice) / K.sum(nb_true_slice)
		#weighted_b_loss = (b_pred_slice * b_true_slice) / K.sum(b_true_slice)
	
		#return weighted_nb_loss + weighted_b_loss
		
	return weighted_xentropy
	
def get_model_loss(batch_size):
	return 'categorical_crossentropy'
	#return batch_weighted_xentropy(batch_size)