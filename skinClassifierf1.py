
import random
import os, cv2
import numpy as np
import pandas as pd
from PIL import Image

from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.models import load_model
from keras.preprocessing.image import  img_to_array


from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import confusion_matrix

K.set_image_dim_ordering('tf')
dataSet = pd.read_csv("Truth Tables/ISIC2018_Task3_Training_GroundTruth.csv") 
image_Names = dataSet["image"].values 
one_hot_arr = dataSet.loc[:,'MEL':'VASC'] 
decoded_images = decode(one_hot_arr)
img_rows, img_cols = 64, 64 
num_channel = 3 
first_epochs = 50
second_epochs = 25
third_epochs = 10
num_categories = 7 
batch_size = 32
num_filters = 32
augmentation_rate = .5
num_transformations = 3
nb_pool=2
nb_conv=5

def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

def decode(datum):
    return np.argmax(np.array(one_hot_arr), axis=1)

x = []
y = []
filepath = 'Images//'
for img in range(int(len(image_Names))): 
        im = Image.open(filepath + image_Names[img] + '.jpg')
        im = im.convert(mode='RGB')
        imrs = im.resize((img_rows, img_cols))
        imrs = img_to_array(imrs)/255
        imrs = imrs.transpose(2,0,1)
        imrs = imrs.reshape(img_rows, img_cols, num_channel)
        #128, 128, 3 shape at this point
        x.append(imrs)
        y.append(decoded_images[img])

x = np.array(x)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)

#Set up the CNN
classifier = Sequential()
classifier.add(Convolution2D(num_filters, nb_conv, nb_conv, input_shape=(img_rows, img_cols, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
classifier.add(Convolution2D(num_filters, nb_conv, nb_conv, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
classifier.add(Convolution2D(num_filters, nb_conv, nb_conv, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(activation="softmax", units = num_categories)) 
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

def print_cnf(x_test, y_test):
    y_pred = classifier.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)

#start training
for first in range(3):
    classifier.fit(x_train, y_train, epochs=first_epochs, batch_size=batch_size)
    classifier.save('f1_model1_' + str(first) +'_3.h5')
    score, acc = classifier.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print_cnf(x_test,y_test)


#increase size of dataset and balance proportions    
new_xt = []
new_yt = []
for x in range(len(x_train)):
    categ = y_train[x]    
    if(categ == 1):
            new_xt.append(x_train[x])
            new_yt.append(y_train[x])
    if(categ == 2 or categ == 3 or categ == 4 or categ == 5):
            new_xt.append(x_train[x])
            new_yt.append(y_train[x])
            transformed1 = random_rotation(x_train[x])
            new_xt.append(transformed1)
            new_yt.append(y_train[x])
            transformed2 = random_noise(x_train[x])
            new_xt.append(transformed2)
            new_yt.append(y_train[x])
            transformed3 = horizontal_flip(x_train[x])
            new_xt.append(transformed3)
            new_yt.append(y_train[x])

new_xt = np.array(new_xt)
new_yt = np.array(new_yt)            

#continue training
for second in range(5):
    classifier.fit(new_xt, new_yt, epochs=second_epochs, batch_size=batch_size)
    classifier.save('f1_model2_' + str(second) +'_3.h5')
    score, acc = classifier.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print_cnf(x_test,y_test)

for third in range(25):
    classifier.fit(new_xt, new_yt, epochs=third_epochs, batch_size=batch_size)
    classifier.save('f1_model3_' + str(third) +'_3.h5')
    score, acc = classifier.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print_cnf(x_test,y_test)


