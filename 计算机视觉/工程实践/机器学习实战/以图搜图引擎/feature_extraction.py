#!/usr/bin/env python
# coding: utf-8

# <table class="tfo-notebook-buttons" align="center">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/practicaldl/Practical-Deep-Learning-Book/blob/master/code/chapter-4/1-feature-extraction.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/practicaldl/Practical-Deep-Learning-Book/blob/master/code/chapter-4/1-feature-extraction.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# 
# 
# This code is part of [Chapter 4 - Building a Reverse Image Search Engine: Understanding Embeddings](https://learning.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html).
# 
# Note: In order to run this notebook on Google Colab you need to [follow these instructions](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=WzIRIt9d2huC) so that the local data such as the images are available in your Google Drive.

# # Feature Extraction
# 
# This notebook is the first among six of the follow along Jupyter Notebook for Chapter 4. We will extract features from pretrained models like VGG-16, VGG-19, ResNet-50, InceptionV3 and MobileNet and benchmark them using the Caltech101 dataset.
# 
# ## Dataset:
# 
# In the `data` directory of the repo, download the Caltech101 dataset (or try it on your dataset). 
# 
# ```
# $ curl http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz --output caltech101.tar.gz
# 
# $ tar -xvzf caltech101.tar.gz
# 
# $ mv 101_ObjectCategories datasets/caltech101
# ```
# Note that there is a 102nd category called ‘BACKGROUND_Google’ consisting of random images not contained in the first 101 categories, which needs to be deleted before we start experimenting. 
# 
# ```
# $ rm -rf datasets/caltech101/BACKGROUND_Google
# ```

# In[3]:


get_ipython().system('mkdir -p ../../datasets')
get_ipython().system('curl http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz --output ../../datasets/caltech101.tar.gz')
get_ipython().system('tar -xvzf ../../datasets/caltech101.tar.gz --directory ../../datasets')
get_ipython().system('mv ../../datasets/101_ObjectCategories ../../datasets/caltech101')
get_ipython().system('rm -rf ../../datasets/caltech101/BACKGROUND_Google')


# In[ ]:


import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import os
import random
import time
import math
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D


# We will define a helper function that allows us to choose any pretrained model with all the necessary details for our experiments.

# In[ ]:


def model_picker(name):
    if (name == 'vgg16'):
        model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3),
                      pooling='max')
    elif (name == 'vgg19'):
        model = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3),
                      pooling='max')
    elif (name == 'mobilenet'):
        model = MobileNet(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3),
                          pooling='max',
                          depth_multiplier=1,
                          alpha=1)
    elif (name == 'inception'):
        model = InceptionV3(weights='imagenet',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='max')
    elif (name == 'resnet'):
        model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                        pooling='max')
    elif (name == 'xception'):
        model = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                         pooling='max')
    else:
        print("Specified model not available")
    return model


# Now, let's put our function to use.

# In[ ]:


model_architecture = 'resnet'
model = model_picker(model_architecture)


# Let's define a function to extract image features given an image and a model. We developed a similar function in Chapter-2

# In[ ]:


def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(img_path,
                         target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features


# Let's see the feature length the model generates. 

# In[ ]:


features = extract_features('../../sample-images/cat.jpg', model)
print(len(features))


# Now, we will see how much time it takes to extract features of one image.

# In[ ]:


get_ipython().run_line_magic('timeit', "features = extract_features('../../sample-images/cat.jpg', model)")


# The time taken to extract features is dependent on a few factors such as image size, computing power etc. A better benchmark would be running the network over an entire dataset. A simple change to the existing code will allow this.
# 
# Let's make a handy function to recursively get all the image files under a root directory.

# In[ ]:


extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def get_file_list(root_dir):
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
    return file_list


# Now, let's run the extraction over the entire dataset and time it.

# In[ ]:


# path to the your datasets
root_dir = '../../datasets/caltech101'
filenames = sorted(get_file_list(root_dir))

feature_list = []
for i in tqdm_notebook(range(len(filenames))):
    feature_list.append(extract_features(filenames[i], model))


# Now let's try the same with the Keras Image Generator functions.

# In[ ]:


batch_size = 64
datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

generator = datagen.flow_from_directory(root_dir,
                                        target_size=(224, 224),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)

num_images = len(generator.filenames)
num_epochs = int(math.ceil(num_images / batch_size))

start_time = time.time()
feature_list = []
feature_list = model.predict_generator(generator, num_epochs)
end_time = time.time()


# In[ ]:


for i, features in enumerate(feature_list):
    feature_list[i] = features / norm(features)

feature_list = feature_list.reshape(num_images, -1)

print("Num images   = ", len(generator.classes))
print("Shape of feature_list = ", feature_list.shape)
print("Time taken in sec = ", end_time - start_time)


# ### GPU Utilization's effect on time taken by varying batch size 
# 
# 
# GPUs are optimized to parallelize the feature generation process and hence will give better results when multiple images are passed instead of just one image.
# The opportunity to improve can be seen based on GPU Utilization. Low GPU Utilization indicates an opportunity to further improve the througput.
# 
# 
# GPU Utilization can be seen using the nvidia-smi command. To update it every half a second
# 
#     watch -n .5 nvidia-smi
#     
# To pool the GPU utilization every second and dump into a file
# 
#     nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -f gpu_utilization.csv -l 1
#     
# To calculate median GPU Utilization from the file generated
# 
#     sort -n gpu_utilization.csv | datamash median 1
# 
# |Model |Time second (sec) | batch_size | % GPU Utilization | Implementation|
# |-|-|-|
# |Resnet50 | 124  | 1  | 52 | extract_features    |
# |Resnet50 | 98   | 1  | 72 | ImageDataGenerator |
# |Resnet50 | 57   | 2  | 81 | ImageDataGenerator |
# |Resnet50 | 40   | 4  | 88 | ImageDataGenerator |
# |Resnet50 | 34   | 8  | 94 | ImageDataGenerator |
# |Resnet50 | 29   | 16 | 97 | ImageDataGenerator |
# |Resnet50 | 28   | 32 | 97 | ImageDataGenerator |
# |Resnet50 | 28   | 64 | 98 | ImageDataGenerator |

# ### Some benchmarks on different model architectures to see relative speeds
# 
# Keeping batch size of 64, benchmarking the different models
# 
# |Model |items/second |
# |-|-|-|
# | VGG19     | 31.06 |
# | VGG16     | 28.16 | 
# | Resnet50  | 28.48 | 
# | Inception | 20.07 |
# | Mobilenet | 13.45 |

# Let's save the features as intermediate files to use later.

# In[ ]:


filenames = [root_dir + '/' + s for s in generator.filenames]


# In[ ]:


pickle.dump(generator.classes, open('./data/class_ids-caltech101.pickle',
                                    'wb'))
pickle.dump(filenames, open('./data/filenames-caltech101.pickle', 'wb'))
pickle.dump(
    feature_list,
    open('./data/features-caltech101-' + model_architecture + '.pickle', 'wb'))


# Let's train a finetuned model as well and save the features for that as well.

# In[ ]:


TRAIN_SAMPLES = 8677
NUM_CLASSES = 101
IMG_WIDTH, IMG_HEIGHT = 224, 224


# In[ ]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2)


# In[ ]:


train_generator = train_datagen.flow_from_directory(root_dir,
                                                    target_size=(IMG_WIDTH,
                                                                 IMG_HEIGHT),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=12345,
                                                    class_mode='categorical')


# In[ ]:


def model_maker():
    base_model = ResNet50(include_top=False,
                           input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=predictions)


# In[ ]:


model_finetuned = model_maker()
model_finetuned.compile(loss='categorical_crossentropy',
              optimizer=tensorflow.keras.optimizers.Adam(0.001),
              metrics=['acc'])
model_finetuned.fit_generator(
    train_generator,
    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES) / batch_size),
    epochs=10)


# In[ ]:


model_finetuned.save('./data/model-finetuned.h5')


# In[ ]:


start_time = time.time()
feature_list_finetuned = []
feature_list_finetuned = model_finetuned.predict_generator(generator, num_epochs)
end_time = time.time()

for i, features_finetuned in enumerate(feature_list_finetuned):
    feature_list_finetuned[i] = features_finetuned / norm(features_finetuned)

feature_list = feature_list_finetuned.reshape(num_images, -1)

print("Num images   = ", len(generator.classes))
print("Shape of feature_list = ", feature_list.shape)
print("Time taken in sec = ", end_time - start_time)


# In[ ]:


pickle.dump(
    feature_list,
    open('./data/features-caltech101-resnet-finetuned.pickle', 'wb'))

