
# coding: utf-8

# In[1]:


import tensorflow as tf
import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# In[2]:


'''use multi image'''
import os
import csv

samples = []
with open('/my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
del samples[0]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #use center camera imgs
                center = '/my_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center)
                center_image= cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])                
                images.append(center_image)
                angles.append(center_angle)
                #flip imgs
                image_flipped=np.fliplr(center_image)
                images.append(image_flipped)
                angles.append(-center_angle)

                #use left camera imgs
                left = '/my_data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left)
                left_image= cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = float(float(batch_sample[3])+0.2)                
                images.append(left_image)
                angles.append(left_angle)    
                #flip imgs
                image_flipped=np.fliplr(left_image)
                images.append(image_flipped)
                angles.append(-left_angle)                
                
                #use right camera imgs
                right = '/my_data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right)
                right_image= cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_angle = float(float(batch_sample[3])-0.2)                
                images.append(right_image)
                angles.append(right_angle) 
                #flip imgs
                image_flipped=np.fliplr(right_image)
                images.append(image_flipped)
                angles.append(-right_angle)
                
                
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

'''bulid nvidia model'''
model= Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)))#normaliztion
model.add(Cropping2D(cropping=((50,20),(0,0))))#crooping
model.add(Convolution2D(24,(5,5),subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,(5,5),subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,(5,5),subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Dropout(0.5))#use dropout
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

from keras.models import Model
import matplotlib.pyplot as plt


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), 
              epochs=3, verbose=1, callbacks=None,
              validation_data=validation_generator, 
              validation_steps=len(validation_samples), class_weight=None, 
              max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)

model.save('/output/model26.h5')

### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.savefig('/output/visual_loss.png')
plt.show()

