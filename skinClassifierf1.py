#import everything
#for data
import os, cv2
import numpy as np
import pandas as pd
from PIL import Image
#from keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential # a keras model that allows sequential convolutional layers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks #Used to view the training process at each stage
#for optimization
from sklearn.utils import shuffle # to shuffle the data before training
from sklearn.cross_validation import train_test_split # cross validation
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import  img_to_array
from keras.models import load_model
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from sklearn.metrics import f1_score

K.set_image_dim_ordering('tf')


#not mineeee
def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]



def decode(datum):
    return np.argmax(np.array(one_hot_arr), axis=1)

print(os.getcwd())

dataSet = pd.read_csv("Truth Tables/ISIC2018_Task3_Training_GroundTruth.csv") 
image_Names = dataSet["image"].values #looks at the csv and reads the image column into an array
one_hot_arr = dataSet.loc[:,'MEL':'VASC'] #needs to be of all of the rows
decoded_images = decode(one_hot_arr)



#CNN Constants
img_rows, img_cols = 64, 64 # to standardize rows and cols
num_channel = 3 # color channels for the image (R, G, and B)
#different numbers of epochs for saving the trained model
first_epochs = 50
second_epochs = 25
third_epochs = 10
num_categories = 7 # or 8? This is the different types of cancer
batch_size = 32
num_filters = 32
augmentation_rate = .5
num_transformations = 3
nb_pool=2
nb_conv=5

#read in images
x = []
y = []
filepath = 'Images//'
#for img in range(int(len(image_Names)/20)):
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


def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0.0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1_score

#Set up the CNN
    #switch this to take in a different number of filters, try 16 and 64
    #add the dropout layers here to minimize overfitting
    #try different activation functions 
#classifier = load_model('my_model_new.h5')
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


##score, acc = classifier.evaluate(x_test, y_test, batch_size=batch_size)
#print('Test score:', score)
#print('Test accuracy:', acc)

def print_cnf(x_test, y_test):
    y_pred = classifier.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)

for first in range(3):
    classifier.fit(x_train, y_train, epochs=first_epochs, batch_size=batch_size)
    classifier.save('f1_model1_' + str(first) +'_3.h5')
    score, acc = classifier.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print_cnf(x_test,y_test)

    
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

#ADD IN A SHUFFLE    
    
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






#Classifier fit generator
