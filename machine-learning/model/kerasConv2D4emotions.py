import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

CSV_FILE = "../dataRetrieval/sources/fer2013.csv"
batch_size = 64
img_width, img_height = 48, 48

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
num_classes = 7


# model_path = '../model/emotion_model.h5'


def _load_fer():
    # Load training and eval data
    df = pd.read_csv(CSV_FILE, sep=',').query('emotion != 1 and emotion !=2 and emotion != 5')
    df.emotion.replace(3, 1, inplace=True)
    df.emotion.replace(4, 2, inplace=True)
    df.emotion.replace(6, 3, inplace=True)
    train_df = df[df['Usage'] == 'Training']
    eval_df = df[df['Usage'] == 'PublicTest']
    return train_df, eval_df


def _preprocess_fer(df,
                    label_col='emotion',
                    feature_col='pixels'):
    labels, features = df.loc[:, label_col].values.astype(np.int32), [
        np.fromstring(image, np.float32, sep=' ')
        for image in df.loc[:, feature_col].values]

    labels = [keras.utils.to_categorical(l, num_classes=num_classes) for l in labels]

    features = np.stack((features,) * 3, axis=-1)
    features /= 255
    features = features.reshape(features.shape[0], img_width, img_height, 3)

    return features, labels


# Load fer data
train_df, eval_df = _load_fer()
num_of_test_samples = eval_df.size

# preprocess fer data
x_train, y_train = _preprocess_fer(train_df)
x_valid, y_valid = _preprocess_fer(eval_df)

print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')

img = x_train[20]
print(img.shape)
plt.imshow(img)
plt.show()

gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
predict_size_train = int(np.math.ceil(len(x_train) / batch_size))

gen = ImageDataGenerator()
valid_generator = gen.flow(x_valid, y_valid, batch_size=batch_size)
predict_size_valid = int(np.math.ceil(len(x_valid) / batch_size))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(256))
model.add(Dropout(0.6))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.summary()

history = model.fit_generator(train_generator,
                    steps_per_epoch=predict_size_train * 3,
                    epochs=200,
                    validation_data=valid_generator,
                    validation_steps=predict_size_valid)

model.save("4emotions_model.h5")



import pandas as pd
from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_valid)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=14)
    plt.yticks(tick_marks, classes, rotation=45, ha='right', fontsize=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Plot
    plt.tight_layout()
    plt.ylim(-0.5, 3.5)
    plt.savefig('4emotions.png')
    plt.show()


cm = confusion_matrix(y_true=np.argmax(y_valid, axis=1),
                      y_pred=np.argmax(y_pred, axis=1))
np.set_printoptions(precision=2)
cm_labels = ['angry', 'happy', 'sad', 'neutral']
plot_confusion_matrix(cm, cm_labels, title='Confusion Matrix')




def print_model_evaluation(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig = plt.gcf()
    fig.savefig('4emotions_acc.png')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig = plt.gcf()
    fig.savefig('4emotions_loss.png')
    plt.show()

print_model_evaluation(history)