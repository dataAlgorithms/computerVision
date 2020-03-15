#!/usr/bin/env python
# coding: utf-8

# <table class="tfo-notebook-buttons" align="center">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/practicaldl/Practical-Deep-Learning-Book/blob/master/code/chapter-4/2-similarity-search-level-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/practicaldl/Practical-Deep-Learning-Book/blob/master/code/chapter-4/2-similarity-search-level-2.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# 
# This code is part of [Chapter 4 - Building a Reverse Image Search Engine: Understanding Embeddings ](https://learning.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html).
# 
# Note: In order to run this notebook on Google Colab you need to [follow these instructions](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=WzIRIt9d2huC) so that the local data such as the images are available in your Google Drive.

# # Similarity Search
# 
# ## Level 2
# 
# We benchmark the algorithms based on the time it takes to index images and locate the most similar image based on its features using the Caltech-101 dataset. We also experiment with t-SNE and PCA.
# 
# ### Understanding the time it takes to index images and locate the most similar image based on its features
# 
# For these experiments we will use the features of the Caltech101 dataset that we read above.
# 
# First, let's choose a random image to experiment with. We will be using the same image for all the following experiments. Note: the results may change if the image is changed.

# In[1]:


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


# In[2]:


filenames = pickle.load(open('data/filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(open('data/features-caltech101-resnet.pickle',
                                'rb'))
class_ids = pickle.load(open('data/class_ids-caltech101.pickle', 'rb'))


# In[3]:


num_images = len(filenames)
num_features_per_image = len(feature_list[0])
print("Number of images = ", num_images)
print("Number of features per image = ", num_features_per_image)


# In[4]:


random_image_index = random.randint(0, num_images)


# ### Standard features
# 
# The following experiments are based on the ResNet-50 features derived from the images of the Caltech101 dataset. 
# 
# ### Standard features + Brute Force Algorithm on one image
# 
# We will be timing the indexing for various Nearest Neighbors algorithms, so let's start with timing the indexing for the Brute force algorithm. While running terminal commands in iPython like the `timeit` command, the variables are not stored in memory, so we need to rerun the same command to compute and store the results in the variable. 

# In[5]:


get_ipython().run_line_magic('timeit', "NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list)")
neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list)


# Now, let's look at the time it takes to search for the nearest neighbors for the selected random image using the trained model with the Brute force algorithm.

# In[6]:


get_ipython().run_line_magic('timeit', 'neighbors.kneighbors([feature_list[random_image_index]])')


# ###  Standard features + k-d Tree Algorithm  on one image
# 
# Now let's turn our attention to the next nearest neighbors algorithm, the k-d tree. Let's time the indexing for the k-d tree algorithm.

# In[7]:


get_ipython().run_line_magic('timeit', "NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(feature_list)")
neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='kd_tree').fit(feature_list)


# Now, time the search for the same random image using the k-d tree trained model.

# In[8]:


get_ipython().run_line_magic('timeit', 'neighbors.kneighbors([feature_list[random_image_index]])')


# ###  Standard features + Ball Tree Algorithm  on one image
# 
# Finally, its time for our last nearest neighbors algorithm - the Ball Tree algorithm. As before, let's calculate the time it takes to train the model.

# In[9]:


get_ipython().run_line_magic('timeit', "NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(feature_list)")
neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='ball_tree').fit(feature_list)


# As before, let's time the search for the Ball Tree model.

# In[10]:


get_ipython().run_line_magic('timeit', 'neighbors.kneighbors([feature_list[random_image_index]])')


# We will increase the number of our test images so that we can experiment with how the scalability of different nearest neighbors algorithms change. Let's choose a random set of 100 or 1000 images to experiment. 
# 
# Note: the results may change if any of the images are changed

# Generate a list of images to do the next set of experiments on.

# In[11]:


random_image_indices = random.sample(range(0, num_images), 1000)
random_feature_list = [
    feature_list[each_index] for each_index in random_image_indices
]


# ### Standard features + Brute Force Algorithm on a set of images
# 
# Time the search for the Brute force algorithm.

# In[12]:


neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list)
get_ipython().run_line_magic('timeit', 'neighbors.kneighbors(random_feature_list)')


# ### Standard features +  k-d Tree Algorithm on a set of images
# 
# Time the search for the k-d tree algorithm.

# In[13]:


neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='kd_tree').fit(feature_list)
get_ipython().run_line_magic('timeit', 'neighbors.kneighbors(random_feature_list)')


# ### Standard features +  Ball Tree Algorithm on a set of images
# 
# Time the search for the Ball Tree algorithm.

# In[14]:


neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='ball_tree').fit(feature_list)
get_ipython().run_line_magic('timeit', 'neighbors.kneighbors(random_feature_list)')


# ### PCA
# 
# Now we have seen the time it takes to index and search using nearest neighbor algorithms on the full feature length. We can use PCA to compress the features and reduce the time. As before we set the number of features intended.

# In[15]:


num_feature_dimensions = 100
num_feature_dimensions = min(num_images, num_feature_dimensions,
                             len(feature_list[0]))


# Train the PCA model with the number of desired feature dimensions.

# In[16]:


pca = PCA(n_components=num_feature_dimensions)
pca.fit(feature_list)
feature_list_compressed = pca.transform(feature_list)
feature_list_compressed = feature_list_compressed.tolist()


# Let's try to understand the importance of each of the resultant features. The numbers displayed below show the relative importance of the first 20 features.

# In[17]:


print(pca.explained_variance_ratio_[0:20])


# Repeat the timing experiments. We use the same random image to experiment.
# Note: the results may change if the image is changed.

# ### PCA + Brute Force Algorithm on one image
# 
# Let's time the indexing for the brute force algorithm.

# In[18]:


get_ipython().run_line_magic('timeit', "NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list_compressed)")
neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list_compressed)


# We will now time the search for the brute force algorithm.

# In[19]:


get_ipython().run_line_magic('timeit', 'neighbors.kneighbors([feature_list_compressed[random_image_index]])')


# ###  PCA + k-d Tree Algorithm  on one image
# 
# Time the indexing for the k-d tree algorithm.

# In[20]:


get_ipython().run_line_magic('timeit', "NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(feature_list_compressed)")
neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='kd_tree').fit(feature_list_compressed)


# Time the search for the k-d tree algorithm.

# In[21]:


get_ipython().run_line_magic('timeit', 'neighbors.kneighbors([feature_list_compressed[random_image_index]])')


# ###  PCA + Ball Tree Algorithm  on one image
# 
# Time the indexing for the ball tree algorithm.

# In[22]:


get_ipython().run_line_magic('timeit', "NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(feature_list_compressed)")
neighbors = NearestNeighbors(
    n_neighbors=5, algorithm='ball_tree').fit(feature_list_compressed)


# Time the search for the ball tree algorithm.

# In[23]:


get_ipython().run_line_magic('timeit', 'neighbors.kneighbors([feature_list_compressed[random_image_index]])')


# We use the same random indices to experiment. Note: the results may change if any of the images are changed.
# 
# Generate a list of images to do the next set of experiments on.

# In[24]:


random_feature_list_compressed = [
    feature_list_compressed[each_index] for each_index in random_image_indices
]


# ### PCA + Brute Force Algorithm on a set of images
# 
# Time the search for the brute force algorithm.

# In[25]:


neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list_compressed)
get_ipython().run_line_magic('timeit', 'neighbors.kneighbors(random_feature_list_compressed)')


# ### PCA + k-d Tree Algorithm on a set of images
# 
# Time the search for the k-d tree algorithm.

# In[26]:


neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='kd_tree').fit(feature_list_compressed)
get_ipython().run_line_magic('timeit', 'neighbors.kneighbors(random_feature_list_compressed)')


# ### PCA + Ball Tree Algorithm on a set of images
# 
# Time the search for the Ball Tree algorithm.

# In[27]:


neighbors = NearestNeighbors(
    n_neighbors=5, algorithm='ball_tree').fit(feature_list_compressed)
get_ipython().run_line_magic('timeit', 'neighbors.kneighbors(random_feature_list_compressed)')


# ### Annoy 
# 
# Make sure you have `annoy` installed. You can install it using pip, `pip3 install annoy`.

# In[29]:


from annoy import AnnoyIndex


# In[30]:


# Time the indexing for Annoy
t = AnnoyIndex(2048)  # Length of item vector that will be indexed
starttime = time.time()
for i in range(num_images):
    feature = feature_list[i]
    t.add_item(i, feature)
endtime = time.time()
print(endtime - starttime)
t.build(40)  # 50 trees
t.save('data/caltech101index.ann')


# ### Annoy on one image 
# 
# Time the search for one image for Annoy.

# In[31]:


u = AnnoyIndex(2048)
get_ipython().run_line_magic('timeit', 'u.get_nns_by_vector(feature_list[random_image_index], 5, include_distances=True)')
indexes = u.get_nns_by_vector(feature_list[random_image_index],
                              5,
                              include_distances=True)


# Helper function to time the search for multiple images for Annoy. Perform the search for the same image multiple times to get an average value.
# 

# In[32]:


def calculate_annoy_time():
    for i in range(0, 100):
        indexes = u.get_nns_by_vector(feature_list[random_image_index],
                                      5,
                                      include_distances=True)


# ### Annoy on a set of images
# 
# Time the search for multiple images for Annoy.

# In[33]:


get_ipython().run_line_magic('time', 'calculate_annoy_time()')


# ### PCA + Annoy
# 
# Now, let's time the indexing for Annoy for the PCA generated features.

# In[34]:


starttime = time.time()
# Length of item vector that will be indexed
t = AnnoyIndex(num_feature_dimensions)

for i in range(num_images):
    feature = feature_list_compressed[i]
    t.add_item(i, feature)
endtime = time.time()
print(endtime - starttime)
t.build(40)  # 50 trees
t.save('data/caltech101index.ann')


# ### PCA + Annoy for one image
# 
# Time the search for one image for Annoy.

# In[35]:


u = AnnoyIndex(num_feature_dimensions)
get_ipython().run_line_magic('timeit', 'u.get_nns_by_vector(feature_list_compressed[random_image_index], 5, include_distances=True)')
indexes = u.get_nns_by_vector(feature_list_compressed[random_image_index],
                              5,
                              include_distances=True)


# Helper function to time the search for multiple images for Annoy. Perform the search for the same image multiple times to get an average value.
# 

# In[36]:


def calculate_annoy_time():
    for i in range(0, 100):
        indexes = u.get_nns_by_vector(
            feature_list_compressed[random_image_index],
            5,
            include_distances=True)


# ### PCA + Annoy on a set of images
# 
# Time the search for multiple images for Annoy.

# In[37]:


get_ipython().run_line_magic('time', 'calculate_annoy_time()')


# ### NMS Lib

# In[44]:


import nmslib


# In[45]:


index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(feature_list_compressed)
index.createIndex({'post': 2}, print_progress=True)


# ### NMS Lib on one image

# In[41]:


# Query for the nearest neighbors of the first datapoint
get_ipython().run_line_magic('timeit', 'index.knnQuery(feature_list_compressed[random_image_index], k=5)')
ids, distances = index.knnQuery(feature_list_compressed[random_image_index],
                                k=5)


# ### NMS Lib on a set of images

# In[42]:


# Get all nearest neighbors for all the datapoint
# using a pool of 4 threads to compute
get_ipython().run_line_magic('timeit', 'index.knnQueryBatch(feature_list_compressed, k=5, num_threads=16)')
neighbors = index.knnQueryBatch(feature_list_compressed, k=5, num_threads=16)


# ### Falconn
# 

# In[46]:


import falconn


# In[48]:


# Setup different parameters for Falonn
parameters = falconn.LSHConstructionParameters()
num_tables = 1
parameters.l = num_tables
parameters.dimension = num_feature_dimensions
parameters.distance_function = falconn.DistanceFunction.EuclideanSquared
parameters.lsh_family = falconn.LSHFamily.CrossPolytope
parameters.num_rotations = 1
parameters.num_setup_threads = 1
parameters.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

# Train the Falconn model
falconn.compute_number_of_hash_functions(16, parameters)


# ### Falconn on a set of images

# In[49]:


dataset = np.array(feature_list_compressed)
a = np.random.randn(8677, 100)
a /= np.linalg.norm(a, axis=1).reshape(-1, 1)
dataset = a

index = falconn.LSHIndex(parameters)
get_ipython().run_line_magic('time', 'index.setup(dataset)')

query_object = index.construct_query_object()
num_probes = 1
query_object.set_num_probes(num_probes)

searchQuery = np.array(feature_list_compressed[random_image_index])
searchQuery = a[0]
get_ipython().run_line_magic('timeit', 'query_object.find_k_nearest_neighbors(searchQuery, 5)')


# ### PCA + Annoy

# In[50]:


# Time the indexing for Annoy for the PCA generated features
starttime = time.time()
# Length of item vector that will be indexed
t = AnnoyIndex(num_feature_dimensions)

for i in range(num_images):
    feature = dataset[i]
    t.add_item(i, feature)
endtime = time.time()
print(endtime - starttime)
t.build(40)  # 50 trees
t.save('data/caltech101index.ann')


# In[51]:


u = AnnoyIndex(num_feature_dimensions)
# Time the search for one image for Annoy
get_ipython().run_line_magic('timeit', 'u.get_nns_by_vector(dataset[random_image_index], 5, include_distances=True)')
indexes = u.get_nns_by_vector(dataset[random_image_index],
                              5,
                              include_distances=True)


# ### Some benchmarks on different algorithms to see relative speeds
# 
# These results lead to the benchmarking of time for indexing and searching on Caltech101. Repeating Level 2 on the Caltech256 features we can benchmark that as well. 
# 
# Benchmarking the different models on Caltech101. (Rounded to nearest integer)
# 
# | Algorithm | Number of features indexed | Time to search 1 image (ms) | Time to search 100 images (ms)  | Time to search 1000 images (ms)  | Time to index (ms)    |
# |-------------|----------------------------|------------------------|---------------------------|---|---|---|
# | Brute force | 2048 | 14 | 38 | 240 | 22 | 
# | k-d tree | 2048 | 16 | 2270 | 24100 | 1020    |
# | Ball tree | 2048 | 15 | 1690 | 17000 | 1090   |
# | PCA + brute force | 100 | 1 | 13 | 135 | 0.334   |
# | PCA + k-d tree | 100 | 1 | 77 | 801 | 20   |
# | PCA + ball tree | 100 | 1 | 80 | 761 |  23   |
# | Annoy | 2048 | 0.16 | 40    | 146 | 1420 |
# | PCA + Annoy | 100 | **.008** | **2.3**   | **20.3** | 109 | 
# 
# 
# Benchmarking the different models on Caltech256. (Rounded to nearest integer)
# 
# 
# | Algorithm | Number of features indexed | Time to search 1 image (ms) | Time to search 100 images (ms)  | Time to search 1000 images (ms)  | Time to index (ms)    |
# |-------------|----------------------------|------------------------|---------------------------|---|---|---|
# | Brute force | 2048 |  16 | 135 |  747  | 23  | 
# | k-d tree | 2048 |  15  | 7400  | 73000 |    4580 |
# | Ball tree | 2048 | 15 | 5940  | 59700 |   4750  |
# | PCA + brute force | 100 | 6.42  | 43.8  | 398  |  1.06   |
# | PCA + k-d tree | 100 |  6.46  | 530  | 5200  |  89.6  |
# | PCA + ball tree | 100 | 6.43  |  601 |  6000 |   104  |
# | Annoy | 2048 | .156  |  41.6  | 166  | 4642  |
# | PCA + Annoy | 100 | **.0076**  |   **2.68** | **23.8**  |  296 | 
