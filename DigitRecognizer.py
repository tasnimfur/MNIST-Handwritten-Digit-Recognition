import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, Flatten,Conv2D, MaxPooling2D

### Load the MNIST  dataset 
# the data is shuffled and split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#- Initially the training dataset contains 60000 images with dimensions as ( 28 * 28 ). We have to reshape it to (28 * 28 * 1)
#- Similarly , test dataset contains 10000 images with dimensions as ( 28 * 28 ). We have to reshape it to (28 * 28 * 1)

output_dim = nb_classes = 10    #10 digits
print('x_train shape: ',x_train.shape)
X_train = x_train.reshape(60000, 28,28,1)
X_test = x_test.reshape(10000, 28,28,1)

# Converting the int type to float type pixels and scaling the pixel values from 0-> 255 to 0-> 1.0.

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Converting the Y_train and Y_test arrays to categorical vectors of size 10 for each image in train and test set.

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# build CNN using keras sequential layers.
    # I have used some extra layers such as Maxpooling layer and dropout layer to regulate the overfitting and add some regularization
    # After passing through some convolutional layers and MaxPool layers , we have to flatten the output and add some Dense layers which will ultimately give us the probabilty of the 10 classes in the ouput layer
    # I have used categorical cross entropy for loss and Adam optimizer 
    #I have also used accuracy as metrics to measure our progress after each epoch

    model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                     input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(optimizer='adam',
            loss=keras.losses.categorical_crossentropy,
            metrics=['accuracy'])
model.summary()

batch_size = 128 
epochs = 10 



#     I tried various combination of Convolutional, MaxPooling , Dropout, Flatten And Dense Layers.
#     I found the above model architecture to be the best working model with best accuracy achieved.
#     I also hypertuned the batch size and number of epochs required for our model 


#      Train on 60000 samples, validate on 10000 samples


trained = model.fit(X_train, Y_train, batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, Y_test))

##### Now, Let's evaluate our model on the test datset and see what accuracy we achieved

loss, acc = model.evaluate(X_test,  Y_test, verbose=2)
print('Test loss:', loss)
print('Test accuracy:', acc)



    
    
    
    
    
