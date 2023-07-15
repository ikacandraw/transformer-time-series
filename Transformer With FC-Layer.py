# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:28:25 2022

@author: Ika Candrawengi
"""

import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(123)
np.random.seed(123)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size)

        # LINEAR LAYER
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        v = values.reshape(N, value_len, self.heads, self.head_dim)
        k = keys.reshape(N, key_len, self.heads, self.head_dim)
        q = query.reshape(N, query_len, self.heads, self.head_dim)
         
        value = self.values(v.float())
        key = self.keys(k.float())
        query = self.queries(q.float())
        
        energy = torch.einsum("nqhd,nkhd->nhqk",query, key)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e28"))

        ## RUMUS ATTENTION ##
        attention = torch.softmax(energy/(self.embed_size ** (1/2)), dim=3)
        out = torch.einsum('nhql,nlhd->nqhd', [attention, value]).reshape(
            N, query_len, self.heads*self.head_dim
        )

        # attention shape :(N, heads, query_len, key_len)
        # values shape : (N, value_len, heads, head_dim)
        # (N, query len, heads, head_dim)

        out = self.fc_out(out)
        return (out)
    


# BLOK TRANSFORMER

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, key, query, mask):
        
        attention = self.attention(values, key, query, mask)

        x = self.dropout(self.norm1(attention.float()*query.float()))
        forward = self.feed_forward(x.float())
        out = self.dropout(self.norm2(forward+x))
        return out

## Positional Encoding ##

import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ENCODER


class Encoder(nn.Module):
    def __init__(self,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length
                 ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.en_pos_encoding = PositionalEncoding(
            embed_size, dropout, max_length)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.en_pos_encoding(x)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

    
class TFmodifikasi(nn.Module):
    def __init__(self,
                 input_window,
                 output_window,
                 embed_size=512, 
                 num_layers=4, 
                 forward_expansion=4, 
                 heads=8, 
                 dropout=0.1, 
                 device="cpu",
                 max_length=500,
                 batch_size = 50,
                 ):
            super(TFmodifikasi, self).__init__()
            self.encoder = Encoder(embed_size,
                               num_layers,
                               heads,
                               device,
                               forward_expansion,
                               dropout,
                               max_length)
        
            self.linear =  nn.Sequential(
                    nn.Linear(embed_size, embed_size//2),
                    nn.ReLU(),
                    nn.Linear(embed_size//2, 1)
                    )

            self.linear2 = nn.Sequential(
                    nn.Linear(input_window, (input_window+output_window)//2),
                    nn.ReLU(),
                    nn.Linear((input_window+output_window)//2, output_window)
                    ) 

    def generate_square_subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

    def forward(self, src, srcmask):
            src = self.encoder(src,src_mask)
            output = self.linear(src)[:,:,0]
            output = self.linear2(output)
            #output = torch.softmax(output,dim=1)
            return output

#%%
## COBA DATA##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## Applying Windows Slidding ##
data = pd.read_csv("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/data/Thesis data/SLC price.csv",encoding='CP949')
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
data["Temp"] = min_max_scaler.fit_transform(data["Temp"].to_numpy().reshape(-1,1))
##BAGI TRAIN TESTING##
train = data[:-12]
data_train = train["Temp"].to_numpy()

test = data[-12:]
data_test = test["Temp"].to_numpy()

newdata = data["Temp"].to_numpy()

import math
from torch.utils.data import DataLoader, Dataset

## Definisikan Input dan Output Windows ##
## Input windows = Data yang digunakan untuk input encoder serta decoder
## output windows = berapa jumlah output yang diinginkan dari decoder 
class windowDataset(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=1):
        #총 데이터의 개수
        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y
        
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len

iw = 24
ow = 12

train_dataset = windowDataset(data_train, input_window=iw, output_window=ow, stride=1)
train_loader = DataLoader(train_dataset, batch_size=250)
#test_dataset = windowDataset(data_test,input_window = iw, output_window = ow, stride =1)
#test_loader = DataLoader(test_dataset,batch_size = 300)

i,batch = next(enumerate(train_loader))
src,trg = batch
src_pad_idx = src.shape[1]
trg_pad_idx = trg.shape[1]
src.size()
trg.size()

src_pad_idx=24
trg_pad_idx=12
## TRAINING DATA
model = TFmodifikasi(24, 12, device="cpu")
lr = 0.005
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas = (0.9,0.98))
model_dir = "C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/new best model/head 8 layer 4 (with fc)"
import os

#%%
device= "cpu"
from tqdm import tqdm
epoch = 100
model.train()
train_loss,val_loss = [],[]
progress = tqdm(range(1,epoch+1))
for i in progress:
    batchloss = 0.0
    for (inputs, outputs) in train_loader:
        optimizer.zero_grad()
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
        result = model(inputs.float().to(device),  src_mask)
        loss = criterion(result, outputs[:,:,0].float().to(device))
        loss.backward()
        res = min_max_scaler.inverse_transform(result.detach().numpy())
        trg = min_max_scaler.inverse_transform(outputs[:,:,0].detach().numpy())
        rmse = np.sqrt(np.mean(pow(res - trg, 2)))
        mape = np.mean(np.abs((trg - res)/trg))*100
        optimizer.step()
        batchloss += loss
    if (i % 100 == 0):
        state = {
             'epoch' : epoch,
             'state_dir': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
        torch.save(state, os.path.join(model_dir, 'model fc epoch-{}.pth'.format(i)))
         #torch.save(model.state_dict(), os.path.join(model_dir, 'model fc epoch-{}.pth'.format(epoch)))
        #torch.save(model.state_dict(), os.path.join(model_dir, 'model fc epoch-{}.pth'.format(epoch)))
    #progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
    train_loss.append(batchloss.cpu().item()/len(train_loader))
    
    
    ##Eval model##
    lossval = 0.0
    x = torch.tensor(data_train[-24:]).reshape(1,-1,1)
    y = torch.tensor(newdata[-13:-1]).reshape(1,-1,1)
    val = torch.tensor(data_test).reshape(1,-1,1)
    model.eval()
    src_mask = model.generate_square_subsequent_mask(x.shape[1]).to(device)
    predictions = model(x, src_mask)
    predictions = predictions[:,:].reshape(-1,1)
    y = y.reshape(-1,1)
    losseval = criterion(predictions,y.float().to(device))
    lossval += losseval
    progress.set_description("loss: {:0.6f}".format(lossval))
    val_loss.append(losseval.cpu().item())

import csv
with open("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/new best model/head 8 layer 4 (with fc)/trainloss.csv",'a') as train:
   writer = csv.writer(train)
   writer.writerow((train_loss))
with open("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/new best model/head 8 layer 4 (with fc)/valloss.csv",'a') as vallos :
    writer = csv.writer(vallos)
    writer.writerow(val_loss)
plt.plot(train_loss, label ="Train Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE loss)")
plt.legend()
#%%

##EVALUATE MODEL##
model15 = torch.load("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/new best model/head 8 layer 4 (with fc)/model fc epoch-100.pth")
model.load_state_dict(model15["state_dir"])
optimizer.load_state_dict(model15["optimizer"])
model.eval()
x = torch.tensor(newdata[359:-13]).reshape(1,-1,1)
y = torch.tensor(newdata[-13:-1]).reshape(1,-1,1)
pred= model(x,y)
pred = pred.reshape(-1,1)
y = y.reshape(-1,1)

real = min_max_scaler.inverse_transform(data_test.reshape(-1,1))
result = min_max_scaler.inverse_transform(pred.detach().numpy())
test = data[-12:]
from datetime import datetime
test["Date"]=pd.to_datetime(test["Date"])

plt.figure(figsize=(20,5))
plt.plot(test["Date"],real, label="real")
plt.plot(test["Date"],result, label="predict")
plt.xlabel("Tahun")
plt.ylabel("Harga SLC (US$ per barrel")
plt.legend()
plt.show()

device ="cpu"
model.train()
for (inputs, outputs) in train_loader:
        optimizer.zero_grad()
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
        result = model(inputs.float().to(device),  src_mask)

result = min_max_scaler.inverse_transform(result.detach().numpy())
result = result.reshape(-1,1)
src = trg[-99:,:,:].detach().numpy()
src = min_max_scaler.inverse_transform((src).reshape(-1,1))
########## PERHITUNGAN RMSE ##############################################
def RMSEval(y_pred, y_true):
    diff=np.subtract(y_true,y_pred)
    square=np.square(diff)
    MSE=square.mean()
    RMSE=np.sqrt(MSE) 
    return RMSE 

RMSEval(result, src)

plt.plot(result)
plt.plot(src)

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
MAPE(src,result)

########## PERHITUNGAN RMSE ##############################################
def RMSEval(y_pred, y_true):
    diff=np.subtract(y_true,y_pred)
    square=np.square(diff)
    MSE=square.mean()
    RMSE=np.sqrt(MSE) 
    return RMSE 

RMSEval(result, real)

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
MAPE(real,result)

###### PREDICT FUTURE ######
def predict_future(eval_model, input,output,steps):
    eval_model.eval() 
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, steps,1):
            input[-ow:] = 0     
            output = output
            fore = eval_model(input,input)                        
    return fore
    
forecast = predict_future(model, input,output,24)
