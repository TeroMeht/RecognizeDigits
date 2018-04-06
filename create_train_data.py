import numpy as np
import function_warehouse
import matplotlib.pyplot as plt
import glob
import pickle



def bring_images():
    train_labels = []
    train_dataset = np.ndarray(shape=(5640,28,28),dtype=np.float32)
    
    i = 0
    for filename in glob.glob(r'C:/Users/tero7/OneDrive/Työpöytä/project/image_database/' + '*.png'): 
         im = function_warehouse.import_image(filename)       
         train_dataset[i,:,:] = im
         i = i+1

    return train_dataset,train_labels


def ask_labels(train_dataset,train_labels):
    
    starting_index = len(train_labels)
    while(starting_index < len(train_dataset)):
        plt.ion()
        plt.imshow(train_dataset[starting_index], cmap='gray')
        
        plt.pause(0.001)
        nb = input('Choose the label:  ')
        
      
        try:
            nb = int(nb)
            if nb > 36 or nb < 0:
                print('Luokka ei voi olla alle 0 tai yli 36!')
                break
        except ValueError:
                print('ValueError')
                break

        
        train_labels.append(nb)        
        starting_index = starting_index+1
        
    return train_labels

    
    

def saveta_datasetti(train_data, train_label):
    
    pickle.dump(train_data, open( "train_data.p", "wb" ) )
    pickle.dump(train_label, open( "train_label.p", "wb" ))
    
    
def get_datasetti():
    train_dataset = pickle.load(open( "train_data.p", "rb" ) )
    train_labels  = pickle.load(open( "train_label.p", "rb" ))
    
    return train_dataset,train_labels

# Run this only when creating totally new set of data and labels
# =============================================================================
# train_dataset,train_labels = bring_images()
# =============================================================================
    

# Load dataset and labels from disk
train_dataset,train_labels= get_datasetti()

train_labels = ask_labels(train_dataset,train_labels)
saveta_datasetti(train_dataset,train_labels)


# =============================================================================
# for i in range(0,37):
#     print(i,train_labels.count(i))
#     train_labels.count(0)
# =============================================================================
    