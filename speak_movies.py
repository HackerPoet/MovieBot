import numpy as np
import time, json
from matplotlib import pyplot as plt
import win32com.client as wincl
from scipy import stats

speak = wincl.Dispatch("SAPI.SpVoice")
voices = [v for v in speak.GetVoices()]
speak.Volume = 100
speak.Rate = 1

SEQ_SIZE = 1
MAX_MOVIE_GEN = 180
MOVE_TRY_STOP = 80

print "Loading Dictionary..."
word_dict = {}
word_list = []
with open('data/words.txt') as fin:
	for w2 in fin:
		assert(w2[-1] == '\n')
		w = w2[:-1]
		assert(w not in word_dict)
		word_dict[w] = len(word_list)
		word_list.append(w)
dict_size = len(word_list)
period_ix = word_dict['.']
print "Dictionary Size: " + str(dict_size)

print "Loading Keras..."
import os, math
os.environ['THEANORC'] = "./cpu.theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__
from keras.initializers import RandomUniform
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, RepeatVector
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.embeddings import Embedding
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Model, Sequential, load_model, model_from_json
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
from keras.utils import plot_model, to_categorical
from keras import backend as K
K.set_image_data_format('channels_first')
import keras
print "Keras: ", keras.__version__

print "Loading Model..."
model = load_model('Model.h5')
weights = model.get_weights()
model_json = json.loads(model.to_json())

first_layer = model_json['config'][0]
first_layer['config']['batch_input_shape'] = (1, 1)
first_layer['config']['input_length'] = 1
for layer in model_json['config']:
	if layer['class_name'] == 'LSTM':
		assert(layer['config']['stateful'] == False)
		layer['config']['stateful'] = True

new_model = model_from_json(json.dumps(model_json))
new_model.set_weights(weights)

def make_movie(vecs, model):
	running_str = np.zeros((MAX_MOVIE_GEN,), dtype=np.int32)
	running_str[:SEQ_SIZE] = vecs[:SEQ_SIZE]
	movie = []
	for i in xrange(MAX_MOVIE_GEN):
		if i < SEQ_SIZE:
			w = word_list[vecs[i]]
			movie.append(w)
			running_str[i] = vecs[i]
			
			if i > 0:
				x_in = np.expand_dims(running_str[i-1:i], 0)
				new_model.predict(x_in, batch_size=1)
		else:
			x_in = np.expand_dims(running_str[i-1:i], 0)
			pk = new_model.predict(x_in, batch_size=1)[0,-1]
			pk *= pk
			pk /= np.sum(pk)
			xk = np.arange(pk.shape[0], dtype=np.int32)
			custm = stats.rv_discrete(name='custm', values=(xk, pk))
			y_out = custm.rvs()
			running_str[i] = y_out
			movie.append(word_list[y_out])
			if movie[-1] == '.' and movie[-2] == '.':
				break
	movie_text = ""
	movie = movie[SEQ_SIZE:]
	for i in xrange(len(movie)):
		w = movie[i]
		if i > MOVE_TRY_STOP and w == '.':
			movie_text += '.'
			break
		elif len(movie_text) == 0 or w in ['.', ',', "'", '!', '?', ':', ';']:
			movie_text += w
		elif len(movie_text) > 0 and movie_text[-1] == "'" and w in ['s', 're', 't', 'll', 've', 'd']:
			movie_text += w
		else:
			movie_text += ' ' + w
	if movie_text[-1] != '.':
		movie_text += '.'
		
	new_model.reset_states()
	return movie_text

while True:
	print "Generating..."
	movie_text = ""
	while len(movie_text) <= 1:
		x_seeds = np.random.randint(len(word_list), size=SEQ_SIZE, dtype=np.int32)
		x_seeds[-1] = period_ix
		movie_text = make_movie(x_seeds, model)

	print movie_text
	print ""
	speak.Speak(movie_text)
	
	time.sleep(1)

print "Done"
