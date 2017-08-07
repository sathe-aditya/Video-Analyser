import cv2
import os
import numpy as np
import subprocess as sp
import pylab
import imageio
from tqdm import tqdm

count = 0

def createImages(path):
	global count
	globalArray = []
	for folder in os.listdir(path):
		if not os.path.isdir(os.path.join(path, folder)):
			continue

		if folder == 'liked':
			label = [0,1]
		else:
			label = [1,0]

		for file in os.listdir(os.path.join(path, folder)):
			if file == '.DS_Store':
				continue
			filename = os.path.join(path, folder, file)
			vid = imageio.get_reader(filename, 'ffmpeg')
			frames = vid._meta['nframes']
			for frame in tqdm(range(frames)):
				if frame % 100 == 0 and not frame == 0:
					try:
						image = vid.get_data(frame)[:,:,1]
						image = cv2.resize(np.array(image), (50,50))
						globalArray.append([np.array(image), np.array(label)])
						count += 1
					except:
						continue
	return globalArray


fileToWrite = createImages('Train/')
fileToWrite = np.array(fileToWrite)
np.save('trainingSet.npy', fileToWrite)

print 'Processed {} rows.'.format(count)