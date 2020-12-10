#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from pyvista import set_plot_theme
set_plot_theme('document')


# 
# # Model 2 - Anticline
# 

# A simple anticline structure. We start by importing the necessary
# dependencies:
# 
# 
# 

# In[2]:


# Importing GemPy
import gempy as gp

import pandas as pd
pd.set_option('precision', 2)


# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# Creating the model by importing the input data and displaying it:
# 
# 
# 

# In[4]:

#data resolution can be changed according to requirement in line 49
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"
geo_data = gp.create_data('fold', extent=[0, 1000, 0, 1000, 0, 1000], resolution=[200, 2, 200],
                          path_o=path_to_data + "model2_orientations.csv",
                          path_i=path_to_data + "model2_surface_points.csv")


# In[5]:


geo_data.get_data().head()


# Setting and ordering the units and series:
# 
# 
# 

# In[6]:


gp.map_stack_to_surfaces(geo_data, {"Strat_Series": ('rock2', 'rock1'), "Basement_Series": ('basement')})



# 

# In[8]:


interp_data = gp.set_interpolator(geo_data, theano_optimizer='fast_compile')


# In[9]:


geo_data.orientations


# In[10]:


sol = gp.compute_model(geo_data)


#this cell generates lithology blocks, n_iterations defines the number of lithology blocks which we want by manipulating the data
#lith_block is a 1d array 
lith_blocks = np.array([])
n_iterations = 700
for i in range(n_iterations):
    Z = np.random.normal(1000, 500, size=2)
    geo_data.modify_surface_points([11,15],[4,9], Z=Z)
    gp.compute_model(geo_data)
    lith_blocks = np.append(lith_blocks, geo_data.solutions.lith_block)
    #plt.imshow(geo_data.solutions.lith_block.reshape(50,2,50)[:, 0, :].T,origin='bottom')
lith_blocks = lith_blocks.reshape(n_iterations, -1)
#plt.imshow(geo_data.solutions.lith_block.reshape(50,2,50)[:, 0, :].T,origin='bottom')


# In[17]:


print(lith_blocks.shape)


# In[18]:


#converting 1D array of a single block from lith_blocks to a 50,2,50 3d array
new_blocks = []
for bloc in lith_blocks:
    block_reshaped = np.reshape(bloc,(200,2,200)) #shape == resolution in ln49
    #print(block_reshaped.shape)
    new_blocks.append(block_reshaped) 

    


# In[19]:


#getting slices from the 3d array we got from lith_blocks(the ones saved in 'new_blocks' list)
slices_from_blocks = []
for arr in new_blocks:
    model_slices = arr[:,1,:]
    #print(model_slices.shape)
    slices_from_blocks.append(model_slices)


# In[36]:

import os       #importing os for path
#using enumerate method to save a list of arrays as images into a directory THIS WORKED OUT FINALLY!!!!!also added make directory to make the code more robust. images are added to the directory called "new_directo"!!!!

directo = "/home/ec131769/Documents/gempy_creat_data/training_data"
new_directo = 'gempy_model_'
path_to_directo = os.path.join(directo,new_directo) 
if not os.path.exists(path_to_directo):
    os.mkdir(path_to_directo)
    print("Directory " , path_to_directo ,  " Created ")
else:    
    print("Directory " , path_to_directo ,  " already exists")

#os.mkdir(path)
for i,im in enumerate(slices_from_blocks): #i is the index of the list, im is the array object inside the list
    
    fig = plt.figure(figsize=(4,4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im, cmap = 'viridis')
    plt.savefig(path_to_directo + f'{i}.png')
    plt.close()
    


# code to read the images from the directory

#import cv2 

#import glob 
#img_dir = r'C:\Users\ASUS\Desktop\gempy_models\training_data'  
#data_path = os.path.join(img_dir,'*g') 
#files = glob.glob(data_path) 
#data = [] 
#for f1 in files: 
   # img = cv2.imread(f1) 
   # print(img.shape)
    #data.append(img) 
#using opencv to write images to a directory using the enumerate but it gives null array as output
#disk_dir = r'C:\Users\ASUS\Desktop\gempy_models\training_data\Newfolder'
#for im,image in enumerate(slices_from_blocks):
    #im = cv2.imread(image)
    #print(image[:4])
    
 #   image = image.astype(np.int8)
    #im = im.astype(np.int8)
  #  print(image[:1])
   # print(image.dtype)
    #cv2.imwrite(r'C:\Users\ASUS\Desktop\gempy_models\training_data' + f'{im}.png', image)


# In[118]:


#using PIL to write images to a directory using the enumerate but it gives ERROR
#from PIL import Image
#disk_dir = r'C:\Users\ASUS\Desktop\gempy_models\training_data'
#for i, image in enumerate(slices_from_blocks):
 #   Image.fromarray(image).save(disk_dir + f"{i}.png")


# In[87]:


#def plotimages(image_arr):
 #  fig, axes= plt.subplots(1,i, figsize=(20,20))
  # for img, ax in zip(image_arr, axes):
   #  ax.imshow(img)
    # plt.tight_layout()
     #plt.imshow(img)


# In[127]:


#using enumerate method to save a list of arrays as images into a directory THIS WORKED OUT FINALLY!!!!!!!!!

