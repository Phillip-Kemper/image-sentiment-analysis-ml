import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

CSV_FILE = "sources/fer2013.csv"
model_path = 'model'
img_width, img_height = 48, 48


# FER_emotions: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# used_emotions: = 0=Happy, 1=Sad, 2=Neutral
def load_fer():
    # Load training and eval data
    df = pd.read_csv(CSV_FILE, sep=',').query('emotion != 0 and emotion != 1 and emotion !=2 and emotion != 5')
    df.emotion.replace(3, 0, inplace=True)
    df.emotion.replace(4, 1, inplace=True)
    df.emotion.replace(6, 2, inplace=True)
    train_df = df[df['Usage'] == 'Training']
    eval_df = df[df['Usage'] == 'PublicTest']
    return train_df, eval_df


train_df, eval_df = load_fer()

with open("sources/fer2013_3_emotions", "w") as fp:
    fp.write(train_df)



print()

print(load_fer())
