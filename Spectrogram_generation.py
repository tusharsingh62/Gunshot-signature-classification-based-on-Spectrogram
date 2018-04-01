
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

nfft = 150
sr = 50000

dataframe = pd.read_csv('/home/administrator/Downloads/Preprocessing/Shockwave_dataset/AK-47_24.6_S_shock1.csv') # Path of extracted signature
y = dataframe.iloc[:,0]
y = np.asarray(y)

specgram(y, NFFT= nfft, Fs= sr, noverlap = nfft/4, pad_to = 300)
