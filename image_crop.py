# -*- coding: utf-8 -*-

from PIL import Image
import numpy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import glob
import os


# Functions
def binarize_image(img_path, threshold):
    """Preprocess the image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
    image = binarize_array(image, threshold)
    return image

def binarize_array(numpy_array, threshold):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 1
            else:
                numpy_array[i][j] = 0
    return numpy_array

def palauta_indexi_vaaka_j(numpy_array):
    pysty_vektori = numpy_array.sum(axis = 0).tolist()
    for indexi in range(len(pysty_vektori)):
        if pysty_vektori[indexi] > 10:           
            return indexi
        
def palauta_indexi_pysty_i(numpy_array):    
    vaaka_vektori = numpy_array.sum(axis = 1).tolist()
    for indexi in range(len(vaaka_vektori)):
        if vaaka_vektori[indexi] > 10:           
            return indexi
        
def center_to_left_corner(numpy_array):
    # Detect the corner
    indj = palauta_indexi_vaaka_j(numpy_array)
    indi = palauta_indexi_pysty_i(numpy_array)        
    temp_array = numpy.delete(numpy_array, numpy.s_[0:indi+1], axis=0)  
    temp_array = numpy.delete(temp_array, numpy.s_[0:indj+1], axis=1) 
    return temp_array        

def siisti_reunat(numpy_array):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][0] > 0:
               numpy_array[i][0] = 0
               
            if numpy_array[0][j] > 0:
               numpy_array[0][j] = 0
              
            if numpy_array[len(numpy_array)-1][j] > 0:
               numpy_array[len(numpy_array)-1][j] = 0   
            
            if numpy_array[i][len(numpy_array[0])-1] > 0:
               numpy_array[i][len(numpy_array[0])-1] = 0   
            
    return numpy_array

def sliding_window(file_destination,temp_array, folder, photo_name):
    
    leveyslista  =  width_of_pictures(temp_array)
    korkeuslista =  height_of_pictures(temp_array)
    valit        =  paljon_siirrytaan(temp_array)
    #ikkunan kokoon alustus
    i_max = korkeuslista[0]
    i_min = 0

    
    for i_siirtyma in range(0, 8):
        j_max = leveyslista[i_siirtyma] #leveys
        j_min = 0
        
        if i_siirtyma > 0:
            i_min = i_max + valit[i_siirtyma-1]
            i_max = i_max + korkeuslista[i_siirtyma]+ valit[i_siirtyma-1]

        for j_siirtyma in range(0, 15):
             
             kuva = siisti_reunat(temp_array[i_min:i_max, j_min:j_max])
             kuva = make_image_right_size(kuva)
             kuvan_nimi = str(folder)+str(photo_name) + str(i_siirtyma)+ str(j_siirtyma) + '.png'
             plt.imsave(file_destination +str(kuvan_nimi), numpy.array(kuva).reshape(28,28), cmap=cm.gray)
                
             j_min = j_max + 2 
             j_max = j_max + leveyslista[j_siirtyma] + 2


def make_image_right_size(imag):
    # 1 niin levitystä tulee oikealle ja 0 niin vasemmalle
    if len(imag[0]) < 28:
        for i in range(28-len(imag[0])):
            imag = increase_width(imag)
                    
    if len(imag) < 28: 
        for i in range(28-len(imag)):
            difference = (28-len(imag))
            if difference % 2 == 0: 
                imag = increase_height(imag, 1)        
            else: 
                imag = increase_height(imag, 0) 
    return imag


# Photos need to be 28x28 for this CNN-model
def increase_width(imag):
    imag = numpy.c_[imag,numpy.zeros(len(imag))]
    return imag
    

def increase_height(imag,side): 
    if side == 0:
        imag = numpy.r_[[imag[0]],imag] 
    if side == 1:
        imag = numpy.r_[imag,[imag[0]]]
    return imag

def width_of_pictures(temp_array):
   # Check what is the width of the picture, returns size 15 list  
    montakonollaa = 0
    lista_leveyksista = []
    
    for kaylapi in range(len(temp_array[2])):
        if temp_array[2][kaylapi] == 0:
             montakonollaa = montakonollaa + 1
        if temp_array[2][kaylapi] == 1 and temp_array[2][kaylapi-1] == 0:
             #print(montakonollaa)
             lista_leveyksista.append(montakonollaa)        
             montakonollaa = 0
    return lista_leveyksista

def height_of_pictures(temp_array):
    
    # Check what is the height of the picture, return size 8 list
    montakonollaa = 0
    lista_korkeuksista = []
    for kaylapi in range(len(temp_array)):  
        if temp_array[kaylapi][0] == 0:
             montakonollaa = montakonollaa + 1
        if temp_array[kaylapi][0] == 1:         
             #print(montakonollaa)                  
             if montakonollaa > 12:
                lista_korkeuksista.append(montakonollaa)  
             montakonollaa = 0  
    return lista_korkeuksista    
     

def paljon_siirrytaan(temp_array):
    
    montakonollaa = 0
    lista_valeista = []
    for kaylapi in range(len(temp_array)):  
        if temp_array[kaylapi][0] == 0:
             montakonollaa = montakonollaa + 1
        if temp_array[kaylapi][0] == 1:         
             #print(montakonollaa)                  
             if montakonollaa < 12:
                lista_valeista.append(montakonollaa+2) 
             montakonollaa = 0
    return lista_valeista
   
    


def run():

    subfolder = []
    subfolder.append('Roulette')
    subfolder.append('Immersive_Roulette')
    subfolder.append('SpeedRoulette')
    subfolder.append('Unibet_Roulette')
    subfolder.append('Unibet_Roulette_A')
    subfolder.append('Unibet_Francais')
    subfolder.append('SvenskRoulette')
    
    
    # Threshold for binarizing the image
    threshold = 70;
    
    
    for i in range(len(subfolder)):
        kansio = subfolder[i]
        
        # Path mihin pilkotut kuvat seivataan
        os.chdir(r'C:/Users/tero7/OneDrive/Työpöytä/project/'+ kansio)
        path = os.getcwd()
        file_destination = r'C:/Users/tero7/OneDrive/Työpöytä/project/image_database/'
        print('Cropping photos from the table: ' + kansio)
        
        for filename in glob.glob(path +'/*.png'): 
           
            head, photo_name = os.path.split(filename)
            
            numpy_array = binarize_image(photo_name, threshold)
            temp_array = center_to_left_corner(numpy_array)
            sliding_window(file_destination,temp_array,kansio,photo_name)
        print('Table ok')
    print('Done')       

run()
