from keras.models import load_model
import os
import librosa
import numpy as np
import random
from librosa.feature import mfcc
from scipy.io import wavfile as wav
import sys

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    custom_loss_value=K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
    return custom_loss_value

model = load_model('model.h5',custom_objects={'contrastive_loss': contrastive_loss})

def Data_ready(file1,file2):
    def extract_features_mfcc(wav_file):
        y,sr = librosa.load(wav_file)
        features = mfcc(y, 22050, n_mfcc=13)
        return features    
    def pad_features_mfcc(features, t_max=200):
        n_mfcc, n_time = features.shape
        padded_features = []
        for n in range(n_mfcc):
            if(n_time > t_max):
                padded_features.append(features[n][:t_max])
            else:
                padded_features.append(np.pad(features[n], (0,t_max - n_time)))
        return np.ravel(padded_features, order='F')
    Left_input =[]
    right_input =[]
    features1 = pad_features_mfcc(extract_features_mfcc(file1))
    features2 = pad_features_mfcc(extract_features_mfcc(file2))
    Left_input.append(features1)
    right_input.append(features2)
    Left_input = np.array(Left_input)
    right_input = np.array(right_input)
    return Left_input,right_input

file1 = sys.argv[1]
file2 = sys.argv[2]

left_test,right_test = Data_ready(file1,file2)
Y_prediction_Test = model.predict([left_test,right_test])

if Y_prediction_Test>=0.524000:
    Y_prediction_Test = 1
else:
    Y_prediction_Test = 0

print(Y_prediction_Test)

