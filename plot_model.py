from load_model import load_model
import sys
from os.path import join
from keras.utils import plot_model

model = load_model(join('models', sys.argv[1]))
plot_model(model, to_file=join('models', sys.argv[1], 'model.png'))