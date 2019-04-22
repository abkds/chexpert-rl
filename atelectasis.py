import csv
import numpy as np
import os
import keras
import Augmentor

from keras.preprocessing import image
from keras import optimizers


model = keras.applications.inception_resnet_v2.InceptionResNetV2(
										include_top=True,
                                        weights=None,
                                        input_shape=(320, 320, 1),
                                        pooling='max',
                                        classes=2)


model = keras.applications.densenet.DenseNet121(
                                        include_top=True,
                                        weights=None,
                                        input_shape=(320, 320, 1),
                                        pooling='max',
                                        classes=2)


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
    data = []
    column_names = []

    with open(filepath, 'rt') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        flag = False
        for row in data_reader:
            if not flag:
                column_names = row
                flag = True
            else:
                data.append(row)

    return column_names, np.array(data)


validate_csv_path = '/floyd/input/chexpert_validation/valid.csv'
_, validate_data = read_data(validate_csv_path)


def preprocess_y_ones(y):
    new_y = np.copy(y)
    new_y[new_y == ''] = 0
    new_y = new_y.astype(float)
    new_y = new_y.astype(int)
    new_y[new_y == -1] = 1
    return new_y


cur_dir = os.getcwd()


def get_validation_data(validate_data, col_num):
	validation_len = len(validate_data)
	X = np.empty((validation_len, 320, 320, 1))
	y = preprocess_y_ones(validate_data[:, col_num])

	filepaths = validate_data[:, 0]

	# Generate data
	for i, path in enumerate(filepaths):
		img_path = os.path.join(cur_dir, path)
		img_path = img_path.replace('home/CheXpert-v1.0-small', 'input/chexpert_validation')
		img = image.load_img(img_path, target_size=(320, 320), grayscale=True)

		X[i,] = image.img_to_array(img) / 255

	return X, keras.utils.to_categorical(y, num_classes=2)


val_data = get_validation_data(validate_data, 13)

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='acc',
        patience=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='chexpert_atelectasis_weights.h5',
        monitor='val_acc',
        save_best_only=True,
    )
]

optimizer = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])


p = Augmentor.Pipeline("/floyd/input/atelectasis/", output_directory='/floyd/home/')


p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.zoom(probability=0.3, min_factor=1.05, max_factor=1.2)
p.flip_left_right(probability=0.7)
p.resize(probability=1.0, width=320, height=320)


batch_size = 16
g = p.keras_generator(batch_size=batch_size)

# training
h = model.fit_generator(
    g,
    steps_per_epoch=len(p.augmentor_images)/batch_size,
    epochs=10,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=val_data
)