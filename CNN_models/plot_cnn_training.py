# coding: utf-8

#Keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd
import numpy as np
#import dask.dataframe as dd
from scipy.signal import convolve
import pickle as pkl
with open('cnn_training_history.pkl', 'rb') as pickle_file:
    history = pkl.load(pickle_file)
epochs=200
kernel=np.array([1,1,1,1,1,1,1,1,1])/9
plt.figure(figsize=(6.2,2.4))
plt.plot(history['acc'], label='Training accuracy')
plt.plot(range(4,epochs-4), convolve(history['acc'], kernel, method='direct',mode='same')[4:-4], label='Training accuracy averaged')
plt.plot(history['val_acc'], label='Validation accuracy')
plt.plot(range(4, epochs-4), convolve(history['val_acc'], kernel, method='direct', mode='same')[4:-4], label='Validation accuracy averaged', color='orange')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
