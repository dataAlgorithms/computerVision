#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import math


# Let's place all the configurations up-front. These can be modified in the future based on the dataset of your choice.

# In[ ]:

TRAIN_DATA_DIR = 'data/train/'
VALIDATION_DATA_DIR = 'data/val/'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64


# ## Load and augment the data
# 
# Colored images usually have 3 channels viz. red, green and blue, each with intensity value ranging from 0 to 255. To normalize it (i.e. bring the value between 0 to 1), we can rescale the image by dividing each pixel by 255. Or, we can use the default `preprocess_input` function in Keras which does the preprocessing for us.

# In[ ]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# Time to load the data from its directories and let the augmentation happen! 
# 
# A few key things to note:
# 
# - Training one image at a time can be pretty inefficient, so we can batch them into groups. 
# - To introduce more randomness during the training process, we’ll keep shuffling the images in each batch.
# - To bring reproducibility during multiple runs of the same program, we’ll give the random number generator a seed value.

# In[ ]:


train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                    target_size=(IMG_WIDTH,
                                                                 IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=12345,
                                                    class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical')


# Now that the data is taken care of, we come to the most crucial component of our training process - the model. We will reuse a CNN previously trained on the ImageNet dataset, remove the ImageNet specific classifier in the last few layers, and replace it with our own classifier suited to our problem. For transfer learning, we’ll ‘freeze’ the weights of the original model, i.e. set those layers as unmodifiable, so only the layers of the new classifier (that we add) can be modified. To keep things fast, we’ll choose the MobileNet model. Don’t worry about the specific layers, we’ll dig deeper into those details in [Chapter 4](https://learning.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html).

# ## Define the model

# In[ ]:


def model_maker():
    base_model = MobileNet(include_top=False,
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


# ## Train and test
# 
# With both the data and model ready, all we have left to do is train the model. This is also known as fitting the model to the data. For training any model, we need to pick a loss function, an optimizer, initial learning rate and a metric. Let's discuss these briefly:
# 
# - **Loss function**: The loss function is the objective being minimized. For example, in a task to predict house prices, the loss function could be the mean squared error.
# - **Optimizer**: This is an optimization algorithm that helps minimize the loss function. We’ll choose `Adam`, one of the fastest optimizers out there.
# - **Learning rate**: This defines how quickly or slowly you update the weights during training. Choosing an optimal learning rate is crucial - a big value can cause the training process to jump around, missing the target. On the other hand, a tiny value can cause the training process to take ages to reach the target. We’ll keep it at 0.001 for now.
# - **Metric**: Choose a metric to judge the performance of the trained model. Accuracy is a good explainable metric, especially when the classes are not imbalanced, i.e. roughly equal in size. Note that this metric is not used during training to maximize or minimize an objective.
# 
# You might have noticed the term `epoch` here. One epoch means a full training step where the network has gone over the entire dataset.  One epoch may consist of several mini-batches.
# 
# Run this program and let the magic begin. If you don’t have a GPU, brew a cup of coffee while you wait. You’ll notice 4 statistics - `loss` and `accuracy` on both the training and validation data. You are rooting for the `val_acc`.

# In[ ]:


model = model_maker()
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['acc'])
model.fit_generator(
    train_generator,
    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=math.ceil(float(VALIDATION_SAMPLES) / BATCH_SIZE))


# On our runs, it took was 5 seconds in the very first epoch to reach 90% accuracy on the validation set, with just 500 training images. Whoa! And by the 10th step, we observe about 97% validation accuracy. That’s the power of transfer learning. 
# 
# Without having the model previously trained on ImageNet, getting a decent accuracy on this task would have taken (1) training time anywhere between a couple of hours to a few days (2) tons of more data to get decent results.
# 
# Before we forget, save the model you trained.

# In[ ]:


model.save('model.h5')


# ## Model Prediction
# 
# Now that you have a trained model, you might eventually want to use it later for your application. We can now load this model anytime and classify an image. The Keras function `load_model`, as the name suggests loads the model. 

# In[ ]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model = load_model('model.h5')


# Now let’s try loading our original sample images and see what results we get.

# In[ ]:


img_path = '../sample-images/dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = expanded_img_array / 255.  # Preprocess the image
prediction = model.predict(preprocessed_img)
print(prediction)
print(validation_generator.class_indices)
