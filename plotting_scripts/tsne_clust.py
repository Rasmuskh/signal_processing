from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

#Keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import optimizers
from keras.models import Model

import keras

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
#import dask.dataframe as dd
from scipy.signal import convolve

window_width=300
waveFormLegth=300
#testset
#df_test = pd.read_parquet('../data/finalData/CNN/test.pq', engine='pyarrow').reset_index()
df_test = pd.read_parquet('../data/finalData/CNN/test10min.pq', engine='pyarrow').reset_index()


def get_samples(df):
    S = np.array([None]*df.shape[0])
    for i in range(0, len(df)):
        S[i] = df.samples[i][int(0.5 + df.cfd_trig_rise[i]/1000)-20: int(0.5 + df.cfd_trig_rise[i]/1000)+waveFormLegth-20].astype(np.float64)
    return S
St = get_samples(df_test)
x_test=np.stack(St)#df_test.samples)
x_test=x_test.reshape(len(x_test), window_width, 1)

model_path='weights-improvement-01-0.79.hdf5'#model_zero.hdf5'
model=keras.models.load_model('../CNN_models/%s'%model_path)
predictions = model.predict(x_test)
df_test['pred'] = predictions

int_layer_model = Model(inputs=model.input, outputs=model.get_layer('flat').output)
params = int_layer_model.predict(x_test)
params = StandardScaler().fit_transform(params)

X_embedded = TSNE(n_components=2, init='pca', n_iter=1000, verbose=2, perplexity=20, method='barnes_hut', learning_rate=10, early_exaggeration=2).fit_transform(params)

# pca = PCA(n_components=30)
# pca.fit(params)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.sum())
# params_new = pca.transform(params)
# params_new = params_new.reshape(params_new.shape[1], params_new.shape[0])
X1=X_embedded.reshape(X_embedded.shape[1], X_embedded.shape[0])[0]
X2=X_embedded.reshape(X_embedded.shape[1], X_embedded.shape[0])[1]
plt.scatter(X1, X2, c=df_test.pred, cmap='viridis')
plt.colorbar()
plt.show()
