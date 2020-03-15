#!/usr/bin/env python
# coding: utf-8

# <table class="tfo-notebook-buttons" align="center">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/practicaldl/Practical-Deep-Learning-Book/blob/master/code/chapter-4/2-similarity-search-level-3.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/practicaldl/Practical-Deep-Learning-Book/blob/master/code/chapter-4/2-similarity-search-level-3.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# 
# This code is part of [Chapter 4 - Building a Reverse Image Search Engine: Understanding Embeddings](https://learning.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html).
# 
# Note: In order to run this notebook on Google Colab you need to [follow these instructions](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=WzIRIt9d2huC) so that the local data such as the images are available in your Google Drive.

# # Similarity Search
# 
# ## Level 3
# 
# So far we experimented with different visualization techniques on the results, t-SNE and PCA on the results. Now we will calculate the accuracies of the features obtained from the pretrained and finetuned models. The finetuning here follows the same finetuning technique we learnt in Chapter 2.

# In[ ]:


import numpy as np
import pickle
from tqdm import tqdm, tqdm_notebook
import random
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import PIL
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')


# For these experiments we will use the same features of the Caltech101 dataset that we were using before.
# 
# Let's utilize the features from the previously trained model.

# In[ ]:


filenames = pickle.load(open('data/filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(open('data/features-caltech101-resnet.pickle',
                                'rb'))
class_ids = pickle.load(open('data/class_ids-caltech101.pickle', 'rb'))

num_images = len(filenames)
num_features_per_image = len(feature_list[0])
print("Number of images = ", num_images)
print("Number of features per image = ", num_features_per_image)


# First, let's make a helper function that calculates the accuracy of the resultant features using the nearest neighbors brute force algorithm.

# In[ ]:


# Helper function to get the classname
def classname(str):
    return str.split('/')[-2]


# Helper function to get the classname and filename
def classname_filename(str):
    return str.split('/')[-2] + '/' + str.split('/')[-1]


def calculate_accuracy(feature_list):
    num_nearest_neighbors = 5
    correct_predictions = 0
    incorrect_predictions = 0
    neighbors = NearestNeighbors(n_neighbors=num_nearest_neighbors,
                                 algorithm='brute',
                                 metric='euclidean').fit(feature_list)
    for i in tqdm_notebook(range(len(feature_list))):
        distances, indices = neighbors.kneighbors([feature_list[i]])
        for j in range(1, num_nearest_neighbors):
            if (classname(filenames[i]) == classname(
                    filenames[indices[0][j]])):
                correct_predictions += 1
            else:
                incorrect_predictions += 1
    print(
        "Accuracy is ",
        round(
            100.0 * correct_predictions /
            (1.0 * correct_predictions + incorrect_predictions), 2))


# ### 1. Accuracy of Brute Force over Caltech101 features

# In[ ]:


# Calculate accuracy
calculate_accuracy(feature_list[:])


# ### 2. Accuracy of Brute Force over the PCA compressed Caltech101 features

# In[ ]:


num_feature_dimensions = 100
pca = PCA(n_components=num_feature_dimensions)
pca.fit(feature_list)
feature_list_compressed = pca.transform(feature_list[:])


# Let's calculate accuracy over the compressed features.

# In[ ]:


calculate_accuracy(feature_list_compressed[:])


# ### 3. Accuracy of Brute Force over the finetuned Caltech101 features

# In[ ]:


# Use the features from the finetuned model
filenames = pickle.load(open('data/filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(
    open('data/features-caltech101-resnet-finetuned.pickle', 'rb'))
class_ids = pickle.load(open('data/class_ids-caltech101.pickle', 'rb'))


# In[ ]:


num_images = len(filenames)
num_features_per_image = len(feature_list[0])
print("Number of images = ", num_images)
print("Number of features per image = ", num_features_per_image)


# In[ ]:


# Calculate accuracy
calculate_accuracy(feature_list[:])


# ### 4. Accuracy of Brute Force over the PCA compressed finetuned Caltech101 features

# In[ ]:


# Perform PCA
num_feature_dimensions = 100
pca = PCA(n_components=num_feature_dimensions)
pca.fit(feature_list)
feature_list_compressed = pca.transform(feature_list[:])


# In[ ]:


# Calculate accuracy over the compressed features
calculate_accuracy(feature_list_compressed[:])


# ### Accuracy 
# 
# These results lead to the accuracy on Caltech101. Repeating Level 3 on the Caltech256 features we get its corresponding accuracy. 
# 
# Accuracy on Caltech101.
# 
# | Algorithm | Accuracy using Pretrained features| Accuracy using Finetuned features | 
# |-------------|----------------------------|------------------------|
# | Brute Force | 87.06 | 89.48 | 
# | PCA + Brute Force | 87.65  |  89.39 |
# 
# 
# Accuracy on Caltech256.
# 
# | Algorithm | Accuracy using Pretrained features| Accuracy using Finetuned features | 
# |-------------|----------------------------|------------------------|
# | Brute Force | 58.38 | 96.01 | 
# | PCA + Brute Force | 56.64  | 95.34|
