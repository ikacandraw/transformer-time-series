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
        #print(value.size())
        
        energy = torch.einsum("nqhd,nkhd->nhqk",query, key)
        #print(energy.size())

        #if mask is not None:
            #energy = energy.masked_fill(mask == 0, float("-1e28"))
        
        ## RUMUS ATTENTION ##
        attention = torch.softmax(energy/(self.embed_size ** (1/2)), dim=3)
        #print(attention.size())
        out = torch.einsum("nhql,nlhd->nqhd", attention, value).reshape(
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
from torch import nn, Tensor
class PositionalEncoder(nn.Module):
  def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        exp_input = torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        div_term = torch.exp(exp_input) # Returns a new tensor with the exponential of the elements of exp_input
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) # torch.Size([58, 512])
        pe = pe.unsqueeze(0).transpose(0, 1) # torch.Size([58, 1, 512])
        # register that pe is not a model parameter
        self.register_buffer('pe', pe)  
  def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """
        add = self.pe[:x.size(1), :].squeeze(1)

        x = x + add
        return x

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
        self.en_pos_encoding = PositionalEncoder(
            dropout, max_length, embed_size)
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

# DECODER BLOCK


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention.float()+x.float()))
        out = self.transformer_block(value, key, query, src_mask)
        return out



class Decoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length, batch_size):
        super(Decoder, self).__init__()
        
        self.device = device
        self.dec_pos_encoding = PositionalEncoder(
            dropout, max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size,ow)
        self.linear =  nn.Sequential(
            nn.Linear(embed_size, embed_size//2),
            nn.ReLU(),
            nn.Linear(embed_size//2, iw)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 
        self.dropout = nn.Dropout(dropout)
        self.linear_mapping = nn.Linear(
            in_features=embed_size,
            out_features=ow
        )
    def forward(self, x, enc_out, src_mask, trg_mask):
        seq_length = x.shape[1]
        x = self.dec_pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        #out = self.linear2(out)        
        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=512, 
                 num_layers=4, 
                 forward_expansion=4, 
                 heads=8, 
                 dropout=0.1, 
                 #device="cpu",
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 max_length=500,
                 batch_size = 50):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        
        self.decoder = Decoder(
                               embed_size, 
                               num_layers, 
                               heads, 
                               forward_expansion, 
                               dropout, 
                               device, 
                               max_length,
                               batch_size)
        
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_mask = trg_pad_idx
        self.device = device
    
    def make_src_mask(self,src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self,trg):
        N = trg.shape[0]
        trg_len = 12
        trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        
        return out
 
    
 
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

class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, stride=1):
        L = y.shape[0]
        num_samples = (L - input_window - output_window) // stride + 1

        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[(start_y):end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i, :], self.y[i,1:]
    def __len__(self):
        return self.len
#%%
iw = 24
ow = 12


import math
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
class TransformerDataset(Dataset):
    def __init__(self, 
        data: torch.tensor,
        indices: list, 
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
        ) -> None:
        super().__init__()
        self.indices = indices
        self.data = data
        print("From get_src_trg: data size = {}".format(data.size()))
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
       start_idx = self.indices[index][0]
       end_idx = self.indices[index][1]
       sequence = self.data[start_idx:end_idx]
       src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len = self.target_seq_len
            )

       return src, trg, trg_y

    def get_src_trg(
        self,
        sequence: torch.Tensor, 
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
        ) -> Tuple [torch.tensor, torch.tensor, torch.tensor]:
        assert len(sequence) == enc_seq_len + target_seq_len
        src = sequence[:enc_seq_len] 
        trg = sequence[enc_seq_len-1:len(sequence)-1]
        assert len(trg) == target_seq_len
        trg_y = sequence[-target_seq_len:]
        assert len(trg_y) == target_seq_len
        return src, trg, trg_y.squeeze(-1)
    
def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
  stop_position = len(data)-1
  subseq_first_idx = 0
  subseq_last_idx = window_size
  indices = []
  while subseq_last_idx <= stop_position:

    indices.append((subseq_first_idx, subseq_last_idx))
            
    subseq_first_idx += step_size
            
    subseq_last_idx += step_size

  return indices

iw = 24
ow = 12
training_indices = get_indices_entire_sequence(
    data=data_train, 
    window_size=iw+ow, 
    step_size=1)
len(training_indices)

training_data = TransformerDataset(
    data=torch.tensor(data_train).float(),
    indices=training_indices,
    enc_seq_len=iw,
    dec_seq_len=ow,
    target_seq_len=ow
    )

training_data = DataLoader(training_data, batch_size=250)

i,batch = next(enumerate(training_data))
src, trg, trg_y = batch
train_dataset = windowDataset(data_train, input_window = iw, output_window=ow)
train_loader = DataLoader(train_dataset, batch_size = 250)
#test_dataset = windowDataset(data_test,input_window = iw, output_window = ow, stride =1)
#test_loader = DataLoader(test_dataset,batch_size = 250)


e, batch = next(enumerate(train_loader))
src, trg, trg_y = batch

trg.size()
trg
trg.size()
def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

trg_y.size()





## MASKING
src_mask = generate_square_subsequent_mask(src.shape[1])
trg_mask = generate_square_subsequent_mask(trg.shape[1])


#e,batch2 = next(enumerate(test_loader))
#input_val, output_val = batch2



## TRAINING DATA
model = Transformer(iw, ow)
lr = 0.005
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas = (0.9,0.98))



model_dir = "C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/new best model/head 8 layer 4"



#%%
import os


#trg
epoch = 100
load_model = False
save_model = True
from tqdm import tqdm
#model.train()
train_loss,val_loss = [],[]
progress = tqdm(range(1,epoch+1))

for i in progress:
    batchloss = 0.0
    model.train()
    for (inputs, outputs,target) in train_loader:
        optimizer.zero_grad()
        result_train= model(inputs,outputs)
        loss = criterion(result_train[:,:-1,:], target.float())
        loss.backward()
        res = result_train[:,:-1,0].reshape(-1,1)
        res = min_max_scaler.inverse_transform(res.detach().numpy())
        src = target.detach().numpy()
        src = min_max_scaler.inverse_transform((src).reshape(-1,1))
        rmse = np.sqrt(np.mean(pow(res - src, 2)))
        mape = np.mean(np.abs((src - res)/src))*100
        optimizer.step()
        batchloss += loss
    if (i % 100 == 0):
        state = {
            'epoch' : epoch,
            'state_dir': model.state_dict(),
            'optimizer' : optimizer.state_dict()}
        torch.save(state, os.path.join(model_dir, 'model fc epoch-{}.pth'.format(i)))
    progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(training_data)))
    train_loss.append(batchloss.item()/len(training_data))
    
    ##Eval model##
    lossval = 0.0
    model.eval()
    #for (x,y) in test_loader :
    x = torch.tensor(data_train[-24:]).reshape(1,-1,1)
    y = torch.tensor(newdata[-13:-1]).reshape(1,-1,1)
    val = torch.tensor(data_test).reshape(1,-1,1)
    predictions = model(x,y)
    val = val.reshape(-1,1)
    predictions= predictions[:,-1,:].reshape(-1,1)
    losseval = criterion(predictions,val)
    lossval += losseval
    progress.set_description("val loss: {:0.6f}".format(losseval))
    val_loss.append(losseval.item())

import csv
with open("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/new best model/head 8 layer 4/trainloss.csv",'a') as train:
   writer = csv.writer(train)
   writer.writerow(train_loss)
with open("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/new best model/head 8 layer 4/valloss.csv",'a') as vallos :
    writer = csv.writer(vallos)
    writer.writerow(val_loss)
plt.plot(train_loss, label ="Train Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE loss)")
plt.legend()

y = torch.tensor(newdata[-13:-1]).reshape(1,-1,1)
predictions[:,0]
result_train.size()

#%%
##EVALUATE MODEL##
model15 = torch.load("C:/Users/Ika Candrawengi/OneDrive - Institut Teknologi Sepuluh Nopember/Dokumen/kuliah/Thesis/new best model/head 8 layer 4/model fc epoch-100.pth")
model.load_state_dict(model15["state_dir"])
optimizer.load_state_dict(model15["optimizer"])
model.eval()
x = torch.tensor(data_train[-24:]).reshape(1,-1,1)
y = torch.tensor(newdata[-13:-1]).reshape(1,-1,1)
val = torch.tensor(data_test).reshape(1,-1,1)
pred= model(x,y)
y = val.reshape(-1,1)
pred = pred[:,-1,:].reshape(-1,1)
real = min_max_scaler.inverse_transform((y.numpy()))
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
src = trg_y[-99:,:,:].detach().numpy()
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


#%%
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

#%%
#########################MANUAL###############################################
embed_size = 256
dropout = 0.1
heads = 8
forward_expansion = 4
num_layers = 4

mask = model.make_src_mask(x)
attention = SelfAttention(512, 8)
h = attention(posen,posen,posen,mask=None)
k = attention(posen,posen,posen,mask = mask)
norm1 = nn.LayerNorm(embed_size)
norm2 = nn.LayerNorm(embed_size)

feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
dropout = nn.Dropout(dropout)


posen.type()

positional = PositionalEncoding(512,0.1,5000)
posen = positional(x.float)
posen = posen.float()
trg_mask.size()

values = nn.Linear(32, 32, bias=False)
keys = nn.Linear(32, 32, bias=False)
queries = nn.Linear(32, 32, bias=False)
fc_out = nn.Linear(8*32, 256)


v = posen.reshape(50, 24, 8, 32)
k = posen.reshape(50, 24, 8, 32)
q = posen.reshape(50, 24, 8, 32)

values

value = values(v.float())
key = keys(k.float())
query = queries(q.float())

energy = torch.einsum("nqhd,nkhd->nhqk",query, key)
energy
embed_size = 256

if mask is not None:
    energy = energy.masked_fill(mask==0, float("-1e28"))



## RUMUS ATTENTION ##
attention = torch.softmax(energy/(embed_size ** (1/2)), dim=3)
out = torch.einsum('nhql,nlhd->nqhd', attention , value)
out.reshape(
    50, 24, 8*32
)

attention.size()
value.size()
out = fc_out(out)

layers = nn.ModuleList(
    [
        TransformerBlock(
            embed_size,
            heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
        )
    for _ in range(num_layers)]
)

out = posen.double()
for layer in layers:
     out = layer(out, out, out, mask)
     
out

N = trg.shape[0]
trg_len = 12
trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(
    N, 1, trg_len, trg_len
        )
trg_mask.size()

src= x
src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
src.size()

encod = Encoder(256,4,8,device,4,0.1,500)

hasilencode = encod(posen,mask)

#%%

max_len = 250
d_model = 512
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0).transpose(0, 1)

pe[:x.size(0),:].size()

x.size()

x = x + pe[:x.size(0), :]