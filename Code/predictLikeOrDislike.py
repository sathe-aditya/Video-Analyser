import cv2
import numpy as np
import os
import pylab
import imageio
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from collections import Counter

LR = 1e-03
DIR_PATH = 'Test'
MODEL_NAME = 'pycon-{}-{}.model'.format(LR, '8actual-conv-basic-newData')
IMG_SIZE = 50


def createImages(path):
	global count
	globalArray = []
	for folder in os.listdir(path):
		if not os.path.isdir(os.path.join(path, folder)):
			continue

		for file in os.listdir(os.path.join(path, folder)):
			if file == '.DS_Store':
				continue

			filename = os.path.join(path, folder, file)
			vid = imageio.get_reader(filename, 'ffmpeg')
			frames = vid._meta['nframes']
			for frame in tqdm(range(frames)):
				if frame % 25 == 0 and not frame == 0:
					try:
						image = vid.get_data(frame)[:,:,1]
						image = cv2.resize(np.array(image), (50,50))
						globalArray.append([np.array(image), filename])
						count += 1
					except:
						continue
	return globalArray



convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('model loaded')

folders = os.listdir(DIR_PATH)
for folder in folders:

	if not os.path.isdir(os.path.join(DIR_PATH, folder)):
			continue
	
	for file in os.listdir(os.path.join(DIR_PATH, folder)):
		if file == '.DS_Store':
			continue
		globalArray = []
		filename = os.path.join(DIR_PATH, folder, file)
		vid = imageio.get_reader(filename, 'ffmpeg')
		frames = vid._meta['nframes']
		for frame in tqdm(range(frames)):
			if frame % 25 == 0 and not frame == 0:
				try:
					image = vid.get_data(frame)[:,:,1]
					image = cv2.resize(np.array(image), (50,50))
					globalArray.append([np.array(image), filename])
					count += 1
				except:
					continue
		outputs = []
		for array in globalArray:
			data = array[0]
			modelOutput = model.predict([data.reshape(IMG_SIZE, IMG_SIZE, 1)])[0]
			if np.argmax(modelOutput) == 1: strLabel = 'liked'
			else: strLabel = 'disliked'
			outputs.append(strLabel)
		result = Counter(outputs).most_common(1)[0]
		print '{} ==> {}'.format(filename, result)
		print Counter(outputs).most_common()