# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:53:14 2018

@author: masters
"""

# Larger CNN for the MNIST Dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import cv2
import os
from sklearn.model_selection import train_test_split
K.set_image_dim_ordering('tf')


# load data
folder_path = 'C:\\Users\\masters\\Desktop\\moving object detection\\Data all\\'
images = []
labels = []
class_label = 0

def load_images_from_folder(folder,class_label):
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder, filename))
		if img is not None:
			#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img,(100,250))
			img = img.reshape(100,250,3)
			images.append(img)
			labels.append(class_label)
	class_label=class_label+1
	return class_label

class_label = load_images_from_folder(folder_path+'keyring',class_label)
class_label = load_images_from_folder(folder_path+'mobile',class_label)
class_label = load_images_from_folder(folder_path+'mouse',class_label)
class_label = load_images_from_folder(folder_path+'stappler',class_label)
Data = np.asarray(images)
Labels = np.asarray(labels)

X_train,X_test,y_train,y_test=train_test_split(Data,Labels,test_size=0.2,random_state=2)

print (X_train.shape)
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding="valid",input_shape=( 100, 250,3), activation='relu'))
	#model.add(Conv2D(32, (3, 3), activation='relu',padding = 'valid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu',padding = 'valid'))
	#model.add(Conv2D(64, (3, 3), activation='relu',padding = 'valid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu',padding = 'valid'))
	#model.add(Conv2D(128, (3, 3), activation='relu',padding = 'valid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
# build the model
model = larger_model()
# Fit the model
model.summary()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Large CNN Accuracy: %.2f%%" % (scores[1]*100))
