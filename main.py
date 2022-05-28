# Fashify: Recommend Fashion in Style

# STEP 3: Aim is to Create a Client-Side Web Application
# We are intended to use Streamlit library, which is a Python Frontend Framework to develop web applications.
# In this main.py we will be following these steps to be done:
# 1. file upload -> save the file
# 2. load the file -> feature extraction from that image file
# 3. Find the Recommendations for that image file
# 4. Show the Recommendations on the client-side

# ----------------------------
# importing the required libraries and modules
import streamlit as st

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

import numpy as np
from numpy.linalg import norm
from numpy import extract, save

from pyparsing import col
from scipy.__config__ import show

import os
from PIL import Image
import pickle

from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors


# Giving a suitable title to our Track 3: Algorithms Project
st.title('Fashify: Recommend Fashion In Style')

# ----------------------------
# importing the ResNet50 Model for extracting features of a  sample image. 
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# ----------------------------
# importing filenames and features list created in app.py file. 
# These lists will be read in read-binary mode as they were created in binary format only.
feature_list = np.array(pickle.load(open('extracted_features.pkl','rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


# ----------------------------
#1 FUNCTION TO UPLOAD FILE
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

#2 FUNCTION TO EXTRACT FEATURES OF THE UPLOADED IMAGE
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis = 0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    
    return normalized_result

#3 FUNCTION TO RECOMMEND IMAGES SIMILAR TO THE UPLOADED IMAGE
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])

    return indices


# -----------------------------
uploaded_file = st.file_uploader("You may choose your image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        #file has been uploaded
        # pass
        # displaying the uploaded file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        
        #extracting features of uploaded file
        features = feature_extraction(os.path.join("uploads",uploaded_file.name), model)
        st.text(features)
        
        #recommendation
        indices = recommend(features, feature_list)
        
        #showing the results
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
        
    else:
        st.header("Some Error occurred in uploading your file.")
        
        
# Completed FINAL STEP i.e. we have taken a fashion related image from the user and extracted its features. Now compared the features of this image with the feature list of all the images in our image dataset. After comparison, we have calculated the nearest distance between the vectors and collected the 5 closest vectors. These closest vectors shows the images which have almost similar features to that of the image uploaded by the user. 

# Hence, our Fashion recommendation system works absolutely fine with optimality. 