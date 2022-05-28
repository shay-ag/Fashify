# Fashify: Recommend Fashion in Style

# Step 2: Aim is to test our recommendation system algorithms
# We will take some sample images in our Sample directory and extract the features of those images with the help of our ResNet50 model.
# This sample image will yield some features. Now we will use algorithm to compare the features with our features_list.  
# We are intended to ensure that everything is working fine till this end.

# ----------------------------
# importing the required libraries and modules

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

import numpy as np
from numpy.linalg import norm

import cv2
import pickle

from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# importing filenames and features list created in app.py file. 
# These lists will be read in read-binary mode as they were created in binary format only.
feature_list = np.array(pickle.load(open('extracted_features.pkl','rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# print(feature_list)
# print((feature_list).shape)
# These above print statements are used for debugging purposes in order to check the correct progress of our imported lists.

# ----------------------------
# importing the ResNet50 Model for extracting features of a  sample image. 
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# ----------------------------
# Extracting Features of a sample image from our made sample_images directory.

sample_img = image.load_img('sample_images/image03.jpg', target_size=(224,224))
sample_img_array = image.img_to_array(sample_img)
expanded_sample_img_array = np.expand_dims(sample_img_array, axis=0)
preprocessed_sample_img = preprocess_input(expanded_sample_img_array)
result = model.predict(preprocessed_sample_img).flatten()
normalized_result = result / norm(result)

# Explanation of the above code:
# Firstly we will load the image from its given path by specifying the target size required for it. This loaded image will be stored in a variable named sample_img.
# Next, we will convert this image to an array and store it in sample_img_array variable. After that, we will expand this newly created array with the help of numpy library and then send it for preprocessing and store the result in a new variable names as preprocessed_sample_img. 
# Next we will do prediction and store the result. We will flatten the result and then normalize the result. This normalization is done with the help of norm function imported from numpy module. At the end we can return the normalized result.
# Now, we can use this normalized result to compare the vector values.


# --------------------------
# Calculating Distance Between Vectors
# We now need to calculate the distances between normalized result (features we calculated for sample image in the section just above) and feature_list (features of all the images in our Image Dataset). 
# We are intended to use Nearest Neighbor Algorithm, which can be imported from sklearn machine learning library for Python programming language.

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])
print(indices) 
# These are the 5 image indices nearest to the sample vector

for file in indices[0]:
    # print(filenames[file])
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (128,128)))
    cv2.waitKey(0)

# Explanation of the above code:
# As we have imported scikit learn library, we will use Nearest Neighbor algorithm from it.

# Since our data is not that much large, thus we can simply use Brute Force Method to find the distance between two vectors.
# The distance we will calculate is the Euclidean Distance. Though, we could have used Cosine Distance too, but after analyzing, Euclidean proves to yield better results. 
# Now, we will give the input data in neighbors' fit inbuilt function. In the 'n_nighbors' parameter we specified 5 as input since we require 5 images that are similar enough to the given image.
# This function will return distances and the indices for those vectors. 
# Lastly we have defined a loop for showing the images using opencv library of python. We got the indices from neigbors function, thus we can use this loop to reach out to those indices and show the images available at those points. 


# Completed STEP 2 i.e. tested our recommendation system and the Nearest Neighbor Algorithm we have used which is working fine and giving apt results according to the given sample inputs.
