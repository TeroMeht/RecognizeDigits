import os 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import function_warehouse
from keras.models import model_from_yaml
#os.chdir(r'C:/Users/tero7/OneDrive/Työpöytä/project/')


def load_model_from_disk(model_name,model_coeff):
    
    # load YAML and create model
    yaml_file = open(model_name, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    
    # load weights into new model
    loaded_model.load_weights(model_coeff)
    print("Loaded model from disk")
    
    #Compiling the model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return loaded_model


def bring_test_images(path_from, image_name):
    
    #Define synthetic data
    test_photos = np.ndarray(shape=(1,28,28),dtype=np.float32)
    

    full_path = str(os.path.join(path_from, image_name)) 
    image = function_warehouse.import_image(full_path)
    
    image = np.array(image, dtype='float32')
    test_photos[0,:,:] = image
    return test_photos


def prep_data_keras(img_data):
    
    #Reshaping data for keras, with tensorflow as backend
    img_data = img_data.reshape(len(img_data),28,28,1)
    
    #Converting everything to floats
    img_data = img_data.astype('float32')
    
    #Normalizing values between 0 and 1
    img_data /= 255
    
    return img_data


# Run this only when first running 
model_name = 'model_both.yaml'
model_coeff = "model_both.h5"
loaded_model = load_model_from_disk(model_name, model_coeff)

# Bring in new image, prepare for keras model and predict values
path_from = r'C:/Users/tero7/OneDrive/Työpöytä/project/image_database'
image_name = 'RouletteP5.PNG314.png'

test_photos = bring_test_images(path_from,image_name)
plt.imshow(test_photos[0], cmap='gray')

test_photos = prep_data_keras(test_photos)
predictions = loaded_model.predict(test_photos)

predicted_labels = []   
predicted_labels.append(np.argmax(predictions))
print('Predicted label: ' + str(predicted_labels[0]))
# =============================================================================
# predicted_labels = []
# for j in range(0,2):
#     predicted_labels.append(np.argmax(predictions[j][0]))
# print("Predicted labels: {}\n".format(predicted_labels))
# =============================================================================
