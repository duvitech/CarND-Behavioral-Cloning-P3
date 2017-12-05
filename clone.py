import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
correction = 0.2

images = []
measurements = []

def load_training_data(location):
    heading = True
    with open('../'+ location +'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if heading:
                heading = False
                continue
            lines.append(line)

    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('\\')[-1]
            current_path = '../'+ location +'/IMG/' + filename
            
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float(line[3])        
            if i == 1:
                measurement = measurement + correction
            if i == 2:
                measurement = measurement - correction
            
            measurements.append(measurement)
            image_flipped = cv2.flip(image,1)
            images.append(image_flipped)
            measurement_inverted = measurement * -1.0
            measurements.append(measurement_inverted)

#load_training_data('training_data')
#Sload_training_data('training_data2')
load_training_data('training_data3')

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, SpatialDropout2D
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

model = Sequential()
''' normalize layer '''
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70, 25), (0,0))))

''' nVidia Model '''
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) 

''' leNet Model 
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1)) '''


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h6')

