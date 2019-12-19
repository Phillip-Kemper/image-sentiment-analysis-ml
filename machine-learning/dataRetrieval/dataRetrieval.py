import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

CSV_FILE = "sources/fer2013.csv"
img_width, img_height = 48, 48


# FER_emotions: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# used_emotions: = 0=Happy, 1=Sad, 2=Neutral
def load_fer(csv):
    # Load training and eval data
    df = pd.read_csv(csv, sep=',').query('emotion != 0 and emotion != 1 and emotion !=2 and emotion != 5')
    df.emotion.replace(3, 0, inplace=True)
    df.emotion.replace(4, 1, inplace=True)
    df.emotion.replace(6, 2, inplace=True)
    train_df = df[df['Usage'] == 'Training']
    eval_df = df[df['Usage'] == 'PublicTest']
    return train_df, eval_df


def writeFerCsvInSeparateCsv():
    train_df, eval_df = load_fer(CSV_FILE)

    #with open("sources/fer2013_3_emotions", "w") as fp:
    #    fp.write(train_df)

    train_df.to_csv("sources/fer2013_adapt_train.csv", index=False)
    eval_df.to_csv("sources/fer2013_adapt_eval.csv", index=False)





