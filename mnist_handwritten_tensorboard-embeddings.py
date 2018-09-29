# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:03:21 2018

@author: seraj
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

PATH = os.getcwd()

#LOG_DIR = PATH + '/mnist-tensorboard/log-1'
LOG_DIR = PATH + '/log-1'

embed_count = 1600
mnist = input_data.read_data_sets(PATH + "/data/")
# mnist.train, mnist.test, mnist.validation
batch_xs, batch_ys = mnist.train.next_batch(1600)
x_test = batch_xs/255
y_test = batch_ys
batch_ys.shape[0]
#x_test.shape[0] # the number of images in embed (sprite.PNG)that is number of rows in csv
#x_test.shape[1] # sqrt is the dim of one image that in one row = 28 * 28 

#logdir = r'logdir'  # you will need to change this!!!
logdir = r'C:\Users\seraj\Desktop\myapppython\NUM_MNIST\logdir'  # you will need to change this!!!
# setup the write and embedding tensor

summary_writer = tf.summary.FileWriter(logdir)
#summary_writer: The summary writer used for writing events. config:
embedding_var = tf.Variable(x_test, name='fmnist_embedding') # embedding variable

config = projector.ProjectorConfig() # use the projector
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name # embedding tensor

embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
embedding.sprite.image_path = os.path.join(logdir, 'sprite.png')
embedding.sprite.single_image_dim.extend([28, 28]) # size of single image in the sprite

projector.visualize_embeddings(summary_writer, config) # configuire projector

# run the sesion to create the model check point

with tf.Session() as sesh:  # tensorflow session
    sesh.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sesh, os.path.join(logdir, 'model.ckpt'))
    
    
# create the sprite image and the metadata file

rows = 28
cols = 28

label = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9'] # mape the numbers lable to its name

sprite_dim = int(np.sqrt(x_test.shape[0])) #dim of  sprite.png is sqrt (number of row in csv)

sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim)) # creat blank(temblate) sprite_image

index = 0
labels = []
for i in range(sprite_dim):
    for j in range(sprite_dim):
        
        labels.append(label[int(y_test[index])]) # get what is image(get its number that is lable in csv) the map this number to lable name 
        
        sprite_image[
            i * cols: (i + 1) * cols,
            j * rows: (j + 1) * rows
        ] = x_test[index].reshape(28, 28) * -1 + 1
        
        index += 1
        
        
with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(labels):
        meta.write('{}\t{}\n'.format(index, label))
        
plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')
plt.show()

