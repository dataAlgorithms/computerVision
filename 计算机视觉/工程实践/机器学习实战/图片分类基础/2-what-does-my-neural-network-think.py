#!/usr/bin/env python
# coding: utf-8

# <table class="tfo-notebook-buttons" align="center">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/practicaldl/Practical-Deep-Learning-Book/blob/master/code/chapter-2/2-class-activation-map-on-video.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/practicaldl/Practical-Deep-Learning-Book/blob/master/code/chapter-2/2-what-does-my-neural-network-think.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# 
# This code is part of [Chapter 2 - What’s in the Picture: Image Classification with Keras](https://learning.oreilly.com/library/view/practical-deep-learning/9781492034858/ch02.html). This notebook will not run on Colab. For Colab, use <a target="_blank" href="https://github.com/practicaldl/Practical-Deep-Learning-Book/blob/master/code/chapter-2/2-colab-what-does-my-neural-network-think.ipynb">chapter-2/2-colab-what-does-my-neural-network-think.ipynb</a> instead.

# # What Does My Neural Network Think?
# 
# In this code sample, we try to understand why the neural network made a particular prediction. We use visualization (a heatmap) to understand the decision-making that is going on within the network. Using color, we visually identify the areas within an image that prompted a decision. “Hot” spots, represented by warmer colors (red, orange, and yellow) highlight the areas with the maximum signal, whereas cooler colors (blue, purple) indicate low signal.

# The `visualization.py` script produces the heatmap for one or more input images, overlays it on the image, and stitches it side-by-side with the original image for comparison. The script accepts arguments for image path or a directory that contains frames of a video.

# ## Visualizing the Heatmap of an Image

# In[ ]:


get_ipython().system('python visualization.py --process image --path ../../sample-images/dog.jpg')


# ![t](./data/dog-output.jpg)
# The right half of the image indicates the “areas of heat” along with the correct prediction of a 'Cardigan Welsh Corgi'.
# 
# Note: As we can see below, the label is different from the labels shown in the book. This is because we use the [VGG-19](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) model in the visualization script, whereas we used the [ResNet-50](https://github.com/KaimingHe/deep-residual-networks) model in the book.
# 
# ![t](./data/cat-output.jpg)

# ## Visualizing the Heatmap of a Video
# 
# Before we can run the `visualization.py` script, we will need to use `ffmpeg` to split up a video into individual frames. Let's create a directory to store these frames and pass its name as an argument into the `ffmpeg` command.

# In[ ]:


get_ipython().system('mkdir kitchen')


# In[ ]:


get_ipython().system('ffmpeg -i kitchen-input.mov -vf fps=25 data/kitchen/thumb%04d.jpg -hide_banner')


# Now let's run the `visualization.py` script with the path of the directory containing the frames.

# In[ ]:


get_ipython().system('python visualization.py --process video --path data/kitchen/')


# Compile a video from those frames using ffmpeg:

# In[ ]:


get_ipython().system('ffmpeg -framerate 25 -i data/kitchen_output/result-%04d.jpg kitchen-output.mp4')


# [![Heatmap Demo Video](./data/kitchen-output/result_0001.jpg)](https://youtu.be/DhMzvbYjkUY "Chapter 2 - Heatmap Demo")

# Perfect! Imagine generating heatmaps to analyze the strong points and shortfalls of your trained model or a pretrained model. Don't forget to post your videos on Twitter with the hashtag [#PracticalDL](https://twitter.com/hashtag/PracticalDL)!
