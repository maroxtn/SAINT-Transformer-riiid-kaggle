import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import pandas as pd
import yaml
import time
import torch.optim as optim
import math
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import train, evaluate, epoch_time
from dataloader import TextLoader
from optimizer import NoamOpt
from model import TransformerModel

import warnings
warnings.filterwarnings("ignore")



with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

batch_size = config["batch_size"]
seq_len = config["seq_len"]

N_EPOCHS = config["n_epochs"]


correct_start_token = config["correct_start_token"]
user_answer_start_token = config["user_answer_start_token"]


#Transformer hyperparameter 
d_model = config["d_model"]

decoder_layers = config["decoder_layers"]
encoder_layers = config["encoder_layers"]

dropout = config["dropout"]
ff_model = d_model*4
att_heads = d_model // 64


#Loading questions, and every question corresponding part
que_data = pd.read_csv( "data/raw/questions.csv")
part_valus = que_data.part.values

unique_ques = len(que_data)


auxiliary = pd.read_pickle("data/processed/training.pickle")
val_group = pd.read_pickle("data/processed/validation.pickle")


trainset = TextLoader(auxiliary)
valset = TextLoader(val_group)

train_loader = torch.utils.data.DataLoader(trainset, shuffle=True,
                          batch_size=batch_size, drop_last=True)  #(seq_len, batch_size)

val_loader = torch.utils.data.DataLoader(valset, shuffle=False,
                        batch_size=batch_size, drop_last=False)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
part_valus = torch.LongTensor(part_valus).to(device)



que_emb_size = unique_ques
model = TransformerModel(que_emb_size, hidden=d_model,part_arr=part_valus, dec_layers=decoder_layers, enc_layers=encoder_layers, dropout=dropout, nheads=att_heads, ff_model=ff_model).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'\nThe model has {count_parameters(model):,} trainable parameters \n d_model:{d_model} \n decode layers: {decoder_layers} \n encoder layers: {encoder_layers} \n \n')



optimizer = NoamOpt(d_model, 1, 4000 ,optim.Adam(model.parameters(), lr=0))

#Since the objective of training is a binary classification
criterion = nn.BCEWithLogitsLoss()



best_roc = 0

for epoch in range(N_EPOCHS):   
    print(f'Epoch: {epoch+1:02} ({best_roc})')

    start_time = time.time()

    train_loss = train(model, optimizer, criterion, train_loader)
    valid_loss, acc, roc = evaluate(model, criterion, val_loader)

    epoch_mins, epoch_secs = epoch_time(start_time, time.time())

    if roc > best_roc:
        best_roc = roc
        torch.save(model.state_dict(), 'models/model_best.torch')

    print(f'Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss: {train_loss:.3f}')
    print(f'Val   Loss: {valid_loss:.3f}    Acc {acc:.3f}   ROC {roc:.3f}')
    
print(best_roc)