import json
import numpy as np
from matplotlib import pyplot as plt

NUM_EPOCHS = 100
NUM_RAND_MOVIES = 10
BATCH_SIZE = 128
LR = 0.001
CONTINUE_TRAIN = False
SEQ_SIZE = 8
MOVIE_GEN_LEN = 64
DATA_SPLITS = 20

print("Loading Dictionary...")
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
print("Dictionary Size: " + str(dict_size))

def plotScores(scores, test_scores, fname, on_top=True):
	plt.clf()
	ax = plt.gca()
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.grid(True)
	plt.plot(scores)
	plt.plot(test_scores)
	plt.xlabel('Epoch')
	plt.tight_layout()
	loc = ('upper right' if on_top else 'lower right')
	plt.draw()
	plt.savefig(fname)

print("Loading Keras...")
import os, math
os.environ['THEANORC'] = "./gpu.theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print("Theano Version: " + theano.__version__)
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, RepeatVector, TimeDistributed
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Convolution1D
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

print("Loading Movies...")
y_all = np.load('data/movies.npy')
print("Loaded " + str(y_all.shape[0]) + " movies.")

x_train = []
y_train = []
for i in xrange(y_all.shape[0]):
	j = SEQ_SIZE
	while j < y_all.shape[1]:
		if y_all[i,j-1] == period_ix and y_all[i,j] == period_ix:
			break
		x_train.append(y_all[i,j-SEQ_SIZE:j])
		y_train.append(y_all[i,j-SEQ_SIZE+1:j+1])
		j += 1

x_train = np.array(x_train, dtype=np.int32)
y_train = np.expand_dims(np.array(y_train, dtype=np.int32), axis=-1)
title_delta = y_all.shape[1] - SEQ_SIZE
print("Created " + str(x_train.shape[0]) + " samples.")

split_ix = x_train.shape[0] / 100
x_test = x_train[:split_ix]
y_test = y_train[:split_ix]
x_train = x_train[split_ix:]
y_train = y_train[split_ix:]

#Shuffle the data
print("Shuffling...")
rng_state = np.random.get_state()
np.random.shuffle(x_train)
np.random.set_state(rng_state)
np.random.shuffle(y_train)

if CONTINUE_TRAIN:
	print("Loading Model...")
	model = load_model('Model.h5')
else:
	print("Building Model...")
	model = Sequential()
	print((None,) + x_train.shape[1:])

	model.add(Embedding(dict_size, 160, input_length=SEQ_SIZE))
	print(model.output_shape)
	
	model.add(LSTM(160, return_sequences=True))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	print(model.output_shape)
	
	model.add(TimeDistributed(Dense(dict_size, activation='softmax')))
	print(model.output_shape)

	model.compile(optimizer=RMSprop(lr=LR), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
	plot_model(model, to_file='model.png', show_shapes=True)

def make_movies(vecs, iters, fname):
	running_str = np.zeros((vecs.shape[0], MOVIE_GEN_LEN), dtype=np.float32)
	running_str[:,:SEQ_SIZE] = vecs[:,:SEQ_SIZE]
	movies = ["" for i in xrange(vecs.shape[0])]
	for i in xrange(MOVIE_GEN_LEN):
		if i < SEQ_SIZE:
			for j in xrange(vecs.shape[0]):
				w = word_list[vecs[j,i]]
				movies[j] += w + ' '
			running_str[:,i] = vecs[:,i]
		else:
			y_out = model.predict(running_str[:,i-SEQ_SIZE:i])[:,-1]
			y_out = np.argmax(y_out, axis=1)
			for j in xrange(vecs.shape[0]):
				movies[j] += word_list[y_out[j]] + ' '
			running_str[:,i] = y_out
	with open(fname, 'a') as fout:
		fout.write('===== ' + str(iters) + ' =====\n')
		for m in movies:
			fout.write(m[:-1] + '\n')

print("Training...")
train_loss = []
test_loss = []
train_acc = []
test_acc = []

x_seeds = x_train[0:NUM_RAND_MOVIES * title_delta:title_delta]
for iters in xrange(NUM_EPOCHS):
	make_movies(x_seeds, iters, 'movies.txt')

	if True:
		split_ix = iters % DATA_SPLITS
		split_size = x_train.shape[0] / DATA_SPLITS
		a_ix = split_ix*split_size
		b_ix = (split_ix + 1)*split_size
		history = model.fit(x_train[a_ix:b_ix], y_train[a_ix:b_ix], batch_size=BATCH_SIZE, epochs=1)
	else:
		history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)

	loss = history.history['loss'][-1]
	acc = history.history['sparse_categorical_accuracy'][-1]
	train_loss.append(loss)
	train_acc.append(acc)
	print("Loss: " + str(loss))
	print("Acc:  " + str(acc))
	
	loss, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
	test_loss.append(loss)
	test_acc.append(acc)
	print("Test Loss: " + str(loss))
	print("Test Acc:  " + str(acc))

	plotScores(train_loss, test_loss, 'Loss.png', True)
	plotScores(train_acc, test_acc, 'Acc.png', False)

	model.save('Model.h5')
	print("Saved")

print("Done")
