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
folder_path = 'C:\\Users\\masters\\Desktop\\moving object detection\\Data\\'
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

class_label = load_images_from_folder(folder_path+'bag',class_label)
class_label = load_images_from_folder(folder_path+'car1',class_label)
class_label = load_images_from_folder(folder_path+'car2',class_label)
class_label = load_images_from_folder(folder_path+'car3',class_label)
class_label = load_images_from_folder(folder_path+'car4',class_label)
class_label = load_images_from_folder(folder_path+'Legs',class_label)
class_label = load_images_from_folder(folder_path+'MrCA1',class_label)
class_label = load_images_from_folder(folder_path+'MrCA1Face',class_label)
class_label = load_images_from_folder(folder_path+'MrCA2',class_label)
class_label = load_images_from_folder(folder_path+'MrCA2back',class_label)
class_label = load_images_from_folder(folder_path+'MrCL2',class_label)
class_label = load_images_from_folder(folder_path+'MrMB1',class_label)
class_label = load_images_from_folder(folder_path+'MrMB1down',class_label)
class_label = load_images_from_folder(folder_path+'MrMB2',class_label)
class_label = load_images_from_folder(folder_path+'MrMB2 with bag',class_label)
class_label = load_images_from_folder(folder_path+'MrOC1',class_label)
class_label = load_images_from_folder(folder_path+'MrRA1',class_label)
class_label = load_images_from_folder(folder_path+'MrRA21',class_label)
class_label = load_images_from_folder(folder_path+'MrRA22',class_label)
class_label = load_images_from_folder(folder_path+'MrSU1',class_label)
class_label = load_images_from_folder(folder_path+'MsCL2',class_label)
class_label = load_images_from_folder(folder_path+'MsOC2',class_label)
class_label = load_images_from_folder(folder_path+'MsSI2',class_label)
class_label = load_images_from_folder(folder_path+'Shadow',class_label)
class_label = load_images_from_folder(folder_path+'SI1_1',class_label)
class_label = load_images_from_folder(folder_path+'SI1_2',class_label)
class_label = load_images_from_folder(folder_path+'SI1_3',class_label)
class_label = load_images_from_folder(folder_path+'object missed',class_label)

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
