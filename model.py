from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, concatenate, Softmax
from keras.regularizers import l2 as L2
from keras.layers import Dropout, Reshape

h, w = 512, 512
lmbda = 1e-5

conv_params = {
	'activation': 'relu',
	'padding': 'same'
	#'kernel_regularizer': L2(lmbda)
}

def conv_block(input_layer, num_filter, pool=True):
	c1 = Conv2D(num_filter, 3, **conv_params)(input_layer)
	c2 = Conv2D(num_filter, 3, **conv_params)(c1)
	layers = [c1, c2]
	
	if pool:
		m = MaxPooling2D((2, 2))(c2)
		layers.append(m)
		
	return layers
	
def deconv_block(input_layer, num_filter, conv_block):
	d1 = Conv2D(num_filter, 2, **conv_params)(input_layer)
	u1 = UpSampling2D((2, 2))(d1)
	concat = concatenate([conv_block[-2], u1])
	c1 = Conv2D(num_filter//2, 3, **conv_params)(concat)
	c2 = Conv2D(num_filter//2, 3, **conv_params)(c1)
	return c2

def create_model(num_classes, num_channels):
	inputs = Input(shape=(h, w, num_channels))
	last_layer = inputs
	num_filter = 8
	
	num_blocks = 4
	conv_blocks = []
	
	for _ in range(num_blocks):
		conv_blocks.append(conv_block(last_layer, num_filter))
		last_layer = conv_blocks[-1][-1]
		num_filter *= 2
	
	conv_blocks.append(conv_block(last_layer, num_filter, pool=False))
	last_layer = conv_blocks[-1][-1]
	#last_layer = Dropout(0.5)(conv_blocks[-1][-1])
	del conv_blocks[-1]
	num_filter = num_filter//2
	
	for _ in range(num_blocks):
		last_layer = deconv_block(last_layer, num_filter, conv_blocks[-1])
		del conv_blocks[-1]
		num_filter = num_filter//2
		
	logits = Conv2D(num_classes, 1, **conv_params)(last_layer)
	sm = Softmax()(logits)
	return Model(inputs=inputs, outputs=sm)
	
if __name__ == '__main__':
	model = create_model(2)
	print(model.summary())