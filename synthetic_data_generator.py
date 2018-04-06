# Here we can generate synthetic datasets from mnist digits
import function_warehouse
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import misc
import pickle
import scipy





def build_synth_data(data,labels,dataset_size):
   
    print('Building the data, wait')
    
    #Define synthetic image dimensions
    synth_img_height = 28
    synth_img_width = 28

    #Define synthetic data
    synth_data = np.ndarray(shape=(dataset_size,synth_img_height,synth_img_width),
                           dtype=np.float32)
    
    #Define synthetic labels
    synth_labels = [] 
    
    #For a loop till the size of the synthetic dataset
    for i in range(0,dataset_size):
        
        #Pick a random number of digits to be in the dataset
        num_digits = 2 

        # Limit generated two digits between interval 10...36
        trigger = None
        
        while trigger != True:
            #Randomly sampling indices to extract digits + labels afterwards
            s_indices = [random.randint(0,len(data)-1) for p in range(0,num_digits)]
            
            if y_train[s_indices[0]] == 1 or y_train[s_indices[0]] == 2 or y_train[s_indices[0]] == 3:
                trigger = True
            
            if trigger == True and y_train[s_indices[0]] == 3 and y_train[s_indices[1]] > 6:
                trigger = None
        
        #stitch images together
        new_image = np.hstack([X_train[index] for index in s_indices])
        
        #stitch the labels together
        new_label =  [y_train[index] for index in s_indices] 
        new_label = str(new_label[0])+ str(new_label[1])
        new_label = int(new_label)
        
        #Loop till number of digits 2, to concatenate blanks images, and blank labels together
        for j in range(0,2-num_digits):
            new_image = np.hstack([new_image,np.zeros(shape=(mnist_image_height,
                                                                   mnist_image_width))])
            
        #Resize image
        new_image = misc.imresize(new_image,(28,28))
        
        #Assign the image to synth_data
        synth_data[i,:,:] = new_image
        
        #Assign the label to synth_data
        synth_labels.append(new_label)


    #Return the synthetic dataset
    return synth_data,synth_labels


def create_data_where_both(X_train, X_test,y_train,y_test, amount_of_train, amount_of_test):
   
    # Train dataset
    single_digits_train = X_train[:amount_of_train,:,:]
    both_traindata = np.concatenate((single_digits_train, X_synth_train), axis=0)
    
    # Test dataset
    single_digits_test = X_test[:amount_of_test,:,:]
    both_testdata = np.concatenate((single_digits_test, X_synth_test), axis=0)
    

    
    # Train set labels
    train_labl = y_train[:amount_of_train]
    both_labels_train = np.concatenate((train_labl, y_synth_train), axis=0)
    
    # Test set labels
    test_labl = y_test[:amount_of_test]
    both_labels_test = np.concatenate((test_labl, y_synth_test), axis=0)
    

    return both_traindata,both_testdata,both_labels_train,both_labels_test


#Setting variables for MNIST image dimensions
nist_image_height = 28
nist_image_width = 28

#Import MNIST data from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Building the training dataset
def binarize_it(input_data):  
    for i in range(0,len(input_data)):    
        function_warehouse.binarize_array(input_data[i,:,:],70)   
    return input_data

X_train = binarize_it(X_train)
X_test = binarize_it(X_test)
#plt.imshow(X_train[1753], cmap='gray')


# Building training dataset
X_synth_train,y_synth_train = build_synth_data(X_train,y_train,20000)

# Building the test dataset
X_synth_test,y_synth_test = build_synth_data(X_test,y_test,700)

# Concate training data single and multiple digits in same array, same with testdata
amount_of_singledigits_train = 2000
amount_of_singledigits_test = 300
(both_traindata,both_testdata,both_labels_train,both_labels_test) = create_data_where_both(X_train, X_test,y_train,y_test, amount_of_singledigits_train, amount_of_singledigits_test)



plt.imshow(both_traindata[random.randint(1,22000)], cmap='gray')

pic = both_traindata[random.randint(1,22000)]

pic =scipy.ndimage.zoom(pic, 0.5, output=None, order=1, mode='constant', cval=0.0, prefilter=True)
plt.imshow(pic, cmap = 'gray')



pickle.dump(both_labels_train, open( "synthetic_train_labels.p", "wb" ) )
pickle.dump(both_traindata, open( "synthetic_train_data.p", "wb" ))
pickle.dump(both_labels_test, open( "synthetic_test_labels.p", "wb" ) )
pickle.dump(both_testdata, open( "synthetic_test_data.p", "wb" ))
