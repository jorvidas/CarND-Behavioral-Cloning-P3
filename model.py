import csv
import numpy as np
import cv2
import random

#Read in the data from the driving log.  Counter can be used to limit
#the number of entries being brought in when testing a change to ensure
#that it runs right.
with open('../CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    data = []
    # counter = 0
    for row in reader:
        # if counter >= 31:
            # break
        data.append(row)
        # counter += 1
        
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#Create training and validation data sets
t_samps, v_samps = train_test_split(data, test_size=0.3)

#Create a generator to batch the data into managable portions.
#This randomly chooses right, left, or center camera image
#as the image and the corresponding driving angle to work 
#with then randomly decides whether to flip that image and
#the corresponding driving angle.
#Yields a numpy array of the features and labels.
def generator(samples, batch_size=64):
    num_samples = len(samples)
    correction = .25
    directory = "../CarND-Behavioral-Cloning-P3/data/IMG/"
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for row in batch_samples:
            
                #Randomly choose camera
                column = random.randint(0,2)
                flip = random.randint(0,1)
                image = cv2.imread(directory + row[column].split('/')[-1])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if column == 0:                           
                    angle = float(row[3])
                elif column == 1:
                    angle = float(row[3]) + correction
                else:
                    angle = float(row[3]) - correction

                #Randomly choose flip or keep the same
                if flip:                    
                    image = np.fliplr(image)
                    angle = -angle

                images.append(image)
                angles.append(angle) 
            
            #Create numpy arrays and pass back
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

#Create the generators for the model to work on
t_gen = generator(t_samps, batch_size=16)
v_gen = generator(v_samps, batch_size=16)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

model = Sequential()

#Preprocess the data
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Model
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(.25))
model.add(Dense(84))
model.add(Dropout(.25))
model.add(Dense(1))
adam = Adam(lr=0.0003, decay=1e-6)

#Train and save the model
model.compile(loss='mse', optimizer=adam)
model.fit_generator(t_gen, samples_per_epoch=len(t_samps),
                    validation_data=v_gen, nb_val_samples=len(v_samps),
                    nb_epoch=8)

model.save('modeld.h5')