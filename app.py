# Fashify: Recommend Fashion in Style

# STEP 1 Aim is to extract Features of all the images in our image dataset.
# This app.py file is supposed to create a pickle file.
# The pickle file will contain all the extracted features of all the images in our Fashion Dataset.  
# The extracted features will then be saved and exported to compare with the features of uploaded image by a user.

# ----------------------------
# importing the required libraries and modules

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# -----------------------------
# Firstly we will create a model for extracting features.
# Our model is ResNet50, which is already trained on ImageNET dataset, hence we will not train it but will simply use it for prediction purpose.

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())
# Explanation of above code:
# This print statement will show the summary of our imported model, which was required for debugging purpose. 
# We are converting the output shape of layer from (None, 7, 7, 2048) to (Global (None, 2048)) by MaxPooling2d layer.
# We will be having Non Trainable parameters: 23,587,712 while Trainable params: 0 as we don not require to train ResNet50 model.  
# At this stage, we have our model ready with us.


# -----------------------------
# Function to Extract features
# This function will take 2 parameters, one is path of the image file and another is model through which we will extract the features.
# At the end this function will return   

def extracting_features(path_of_img, model):
    
    new_img = image.load_img(path_of_img, target_size=(224,224))
    new_img_array = image.img_to_array(new_img)
    expanded_new_img_array = np.expand_dims(new_img_array, axis = 0)
    preprocessed_new_img = preprocess_input(expanded_new_img_array)
    resultant = model.predict(preprocessed_new_img).flatten()
    normalized_resultant = resultant / norm(resultant)
    return normalized_resultant

# print(os.listdir('images'))
# Explanation of above code:
# Firstly we will load the image from its given path by specifying the target size required for it. This loaded image will be stored in a variable named new_img.
# Next, we will convert this image to an array and store it in new_img_array variable. After that, we will expand this newly created array with the help of numpy library and then send it for preprocessing and store the result in a new variable names as preprocessed_new_img. 
# Next we will do prediction and store the result. We will flatten the result and then normalize the result. This normalization is done with the help of norm function imported from numpy module. At the end we can return the normalized result.
# Now, we can use this function to send image path and extract features of that image.


# --------------------------------
# Making a Python list that contains all the filenames of the 45k images stored in our images directory. For this we would require os module
filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))
    
# print(len(filenames))
# print(filenames[0:13])
# These print statements are used for debugging purposes in order to check the progress of our newly created filenames list.

# Explanation of the above code:
# Firstly, we have created an empty list named as filenames.
# Then we ran a loop which will append the filenames list to store all the names of image files contained in our image dataset. For running this loop we have used a python module names os. This module will help us to join images string with the file names, i.e. 'images\\1163.jpg
# Now, we have a list that contains all filenames. 


# ------------------------------
# Making a 2D Feature List that will contain features for all the images in our image dataset. 
feature_list = []

for file in tqdm(filenames):
    feature_list.append(extracting_features(file, model))
    
# print(np.array(feature_list).shape)
# This print statement is used for debugging purposes in order to check the progress of our newly created feature list.

# Explanation of the above code:
# Firstly, we will create an empty python list for features_list.
# Then we will run a loop to append the extracted features into that list. These extracted features are taken from the result obtained from previously created function named extracting_features.
# We have used a tqdm module here, in order to track the progress of our running prediction.


# -----------------------------
# Storing the extracted features in a pickle file so that we can use the extracted features of comparison further.

pickle.dump(feature_list, open('extracted_features.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

# we have imported pickle library to export the .pkl files which will be in write binary mode.
# After execution of this code, the pickle files are ready to be exported further, so that we can use it further.


# Completed STEP 1 i.e. Feature Extraction of all the images in our image dataset is now completed and in ready to go mode.
