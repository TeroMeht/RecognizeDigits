#Module Imports
from __future__ import print_function
import random

import numpy as np
from scipy import misc
import pickle
from keras.datasets import mnist
from keras.utils import np_utils
from keras import optimizers


# Selfmade functions
from cropped_image import binarize_array
from keras.utils.np_utils import to_categorical
from make_prediction import load_model_from_disk
import matplotlib.pyplot as plt



def get_datasetti():
    train_dataset = pickle.load(open( "train_data.p", "rb" ) )
    train_labels  = pickle.load(open( "train_label.p", "rb" ))
    
    return train_dataset,train_labels



#-----------------------Second section, prepare for model----------------------------------



def prep_data_keras(img_data):
    
    #Reshaping data for keras, with tensorflow as backend
    img_data = img_data.reshape(len(img_data),28,28,1)
    
    #Converting everything to floats
    img_data = img_data.astype('float32')
    
    #Normalizing values between 0 and 1
    img_data /= 255
    
    return img_data




#---------------------Section three, building the model-------------------------


#Importing relevant keras modules
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

def create_and_train_CNN_model(train_images,train_labels,test_images,test_labels):
    #Building the model
    
    batch_size = 4
    number_classes = 37
    nb_epoch = 30
    
    #image input dimensions
    img_rows = 28
    img_cols = 28
    img_channels = 1
    
    #number of convulation filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    
    #defining the input
    inputs = Input(shape=(img_rows,img_cols,img_channels))
    
    #Model taken from keras example. Worked well for a digit, dunno for multiple
    cov = Convolution2D(nb_filters,kernel_size[0],kernel_size[1],border_mode='same')(inputs)
    cov = Activation('relu')(cov)
    cov = Convolution2D(nb_filters,kernel_size[0],kernel_size[1])(cov)
    cov = Activation('relu')(cov)
    cov = MaxPooling2D(pool_size=pool_size)(cov)
    
    cov = Dropout(0.25)(cov)
    cov_out = Flatten()(cov)
    
    
    #Dense Layers
    cov2 = Dense(128, activation='relu')(cov_out)
    cov2 = Dropout(0.5)(cov2)
    
    
    #Prediction layers
    c10 = Dense(number_classes, activation='softmax')(cov2)

    
    #Defining the model
    model = Model(input=inputs,output=[c10])
        
    #Compiling the model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    

    #Fitting the model
    model.fit(train_images,train_labels,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,
              validation_data=(test_images, test_labels))

    # Save the model to YAML-format--------------------------------------------
    
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model_both.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        
    # serialize weights to HDF5
    model.save_weights("model_both.h5")
    print("Saved model to disk")
    #--------------------------------------------------------------------------



(dataset,labels) = get_datasetti()

trainset = dataset[0:1500,:,:]
testset = dataset[1500:2000,:,:]

trainlabel = labels[0:1500]
testlabel =labels[1500:2000]

# Prepare data for model
train_images = prep_data_keras(trainset)
test_images = prep_data_keras(testset)

trainlabel = np.asarray(trainlabel)
testlabel = np.asarray(testlabel)

# Prepare labels for model
train_labels = to_categorical(trainlabel)
test_labels = to_categorical(testlabel)



# Here if need to train new model
create_and_train_CNN_model(train_images,train_labels,test_images,test_labels)




# Load model from disk
model_name = 'model_both.yaml'
model_coeff = "model_both.h5"
loaded_model = load_model_from_disk(model_name, model_coeff)
predictions = loaded_model.predict(test_images)


def calculate_acc(predictions,real_labels):
    
    individual_counter = 0

    for i in range(0,len(predictions)):
        if np.argmax(predictions[i]) == np.argmax(real_labels[i]):
            individual_counter += 1
         
    ind_accuracy = individual_counter/len(predictions)

    return ind_accuracy

ind_acc = calculate_acc(predictions,test_labels)

print("The model accuracy with test data is {} %".format(ind_acc*100))

