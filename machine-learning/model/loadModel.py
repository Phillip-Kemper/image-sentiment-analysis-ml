# load and evaluate a saved model
from keras.models import load_model
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load model
model = load_model('fer2013_model_2.h5')
print('loaded_model')
# summarize model.
model.summary()
# load dataset
filepath = 'assets/emotest/'
for filename in os.listdir(filepath):
    img = Image.open(os.path.join(filepath, filename))
    img = img.resize((48, 48))
    img = np.copy(np.asarray(img)).astype(float)
    y = 0
    while (y < len(img)):
        x = 0
        while (x < len(img[y])):
            rgb = 0
            addvals = 0
            while (rgb < len(img[y][x])):
                img[y][x][rgb] = img[y][x][rgb] / 255
                # auswertung hiermit: 3,6,6,3,0,3,3,3,6
                # addvals+=img[y][x][rgb]/255
                rgb += 1
            # addvals/=3
            # img[y][x][0]=addvals
            # img[y][x][1]=addvals
            # img[y][x][2]=addvals
            # auswertung hiermit (grau): 3,6,6,3,2,6,3,3,3,0
            x += 1
        y += 1
    plt.imshow(img)
    plt.show()
    print(filename)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    print(np.argmax(prediction))
# 0=Angry, 1=Happy, 2=Sad, 3=Neutral

