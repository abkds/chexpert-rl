import os
import keras
import csv
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

WIDTH = 320
HEIGHT = 320
num_classes = 4

train_csv_path = '/floyd/input/chexpert/train.csv'
validate_csv_path = '/floyd/input/chexpert'


def read_data(filepath):
	"""
	Reads and returns the data from csv file.

	Assumes that the first row is column names.

	Parameters
	----------
	filepath : str
		File path of the file to be read

	Returns
	-------
	(column_names, data)
		Tuple of names of the columns and the data.
	"""
	_data = []
	_column_names = []

	with open(filepath, 'rt') as csvfile:
		data_reader = csv.reader(csvfile, delimiter=',')
		flag = False
		for row in data_reader:
			if not flag:
				_column_names = row
				flag = True
			else:
				_data.append(row)

	return _column_names, _data


column_names, data = read_data(train_csv_path)


def preprocess_y(raw_y):
	raw_y[raw_y == ''] = -2
	raw_y = raw_y.astype(float)
	y = raw_y.astype(int)
	return y


class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self,
				 list_IDs,
				 labels,
				 batch_size=32,
				 dim=(320, 320),
				 n_channels=1,
				 n_classes=10,
				 shuffle=True):
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# generates the indexes of the batch
		indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

		# find the list of ids
		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		# print(list_IDs_temp)

		# data generation
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == False:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size), dtype=int)

		cur_dir = os.getcwd()

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample

			img_path = os.path.join(cur_dir, ID)
			img_path = img_path.replace('home/CheXpert-v1.0-small', 'input/chexpert')
			img = image.load_img(img_path, target_size=(WIDTH, HEIGHT), color_mode='grayscale')

			X[i, ] = image.img_to_array(img) / 255

			# store the class
			y[i] = self.labels[ID]

		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


params = {
	'dim': (320, 320),
	'batch_size': 32,
	'n_classes': 4,
	'n_channels': 1,
	'shuffle': True
}


def get_train_and_validation(train_ids, y_train, validation_ids, y_validation):
	partition = {}
	partition['train'] = train_ids
	partition['validation'] = validation_ids

	labels_ = {}
	for id_, y_ in list(zip(train_ids, y_train)):
		labels_[id_] = y_

	for id_, y_ in list(zip(validation_ids, y_validation)):
		labels_[id_] = y_

	return partition, labels_


if __name__ == '__main__':
	print('Hello world! And bye')



