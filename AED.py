# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:08:55 2022

@author: Ika Candrawengi
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import time
import math
#!pip install pytorch-lightning
torch.manual_seed(123)
np.random.seed(123)

## IMPORT DATA ##
##READ DATA##
data = pd.read_csv("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/data/Thesis data/SLC price.csv",
                   parse_dates=["Date"], 
                   index_col=["Date"], 
                   infer_datetime_format=True,
                   low_memory=False)
data.sort_values("Date",inplace=True)

## AED ##
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
plt.plot(data["Temp"])
plt.xlabel("Tahun")
plt.ylabel("Harga SLC (US$/barrel)")
decompose = seasonal_decompose(data["Temp"],period = 12)
decompose.plot();


fig,ax = plt.subplots(figsize = (20,8))
kk = sns.boxplot(data=data,x=data.index.year,y="Temp",ax=ax, boxprops=dict(alpha=0.3));
sns.swarmplot(data=data,x=data.index.year,y="Temp")
kk.set(xlabel="Tahun",ylabel="Harga SLC (US$/barrel)")



##Boxplot per bulan
import seaborn as sns
data = pd.read_csv("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/data/Thesis data/data.csv")
sns.set_style("whitegrid")
fig,ax =plt.subplots(figsize = (20,8))
bb = sns.boxplot(x=data.index.month,y="Temp",data = data, ax=ax)
sns.swarmplot(data=data,x=data.index.month,y="Temp")
bb.set(xlabel="Bulan",ylabel="Harga SLC (US$/barrel)")

data = pd.read_csv("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/data/Thesis data/SLC price.csv",encoding='CP949')
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
data["Temp"] = min_max_scaler.fit_transform(data["Temp"].to_numpy().reshape(-1,1))
##BAGI TRAIN TESTING##
train = data[:-12]
data_train = train["Temp"].to_numpy()

test = data[-12:]
data_test = test["Temp"].to_numpy()

newdata = min_max_scaler.inverse_transform(data["Temp"].to_numpy().reshape(-1,1))
data = pd.read_csv("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/data/Thesis data/data.csv")
data["Date"]=pd.to_datetime(data["Date"])                                         
plt.plot(data["Date"],data["Temp"],label="Aktual")
plt.plot(data["Date"],newdata,label="MinMaxScaler")
plt.xlabel("Tahun")
plt.ylabel("Harga SLC (US$/barrel)")
plt.legend()

