import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import math
import numpy as np



#Annotated transformers: https://nlp.seas.harvard.edu/2018/04/03/attention.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)




class TransformerModel(nn.Module):
    
    def __init__(self, intoken, hidden, part_arr, enc_layers, dec_layers, dropout, nheads, ff_model, ts_unique=70):
        super(TransformerModel, self).__init__()
        
        self.encoder = nn.Embedding(intoken, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Embedding(3, hidden)  #0: False , 1: Correct , 3 : Padding
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        
        
        self.transformer = nn.Transformer(d_model=hidden, nhead=nheads, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=ff_model, dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(hidden, 1)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
      
        self.part_embedding = nn.Embedding(7,hidden)
        self.part_arr = part_arr
        
        self.ts_embedding = nn.Embedding(ts_unique, hidden)        
        self.user_answer_embedding = nn.Embedding(5, hidden)

        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        self.dropout_5 = nn.Dropout(dropout)
        self.dropout_6 = nn.Dropout(dropout)

        
    def generate_square_subsequent_mask(self, sz, sz1=None):
        
        if sz1 == None:
            mask = torch.triu(torch.ones(sz, sz), 1)
        else:
            mask = torch.triu(torch.ones(sz, sz1), 1)
            
        return mask.masked_fill(mask==1, float('-inf'))


    def forward(self, src, trg, ts, user_answer):

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)
            
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = self.generate_square_subsequent_mask(len(src)).to(trg.device)
            
        if self.memory_mask is None or self.memory_mask.size(0) != len(trg) or self.memory_mask.size(1) != len(src):
            self.memory_mask = self.generate_square_subsequent_mask(len(trg),len(src)).to(trg.device)
            

            
        #Get part, prior, timestamp, task_container and user answer embedding
        part_emb = self.dropout_1(self.part_embedding(self.part_arr[src]-1))
        ts_emb = self.dropout_3(self.ts_embedding(ts))
        user_answer_emb = self.dropout_4(self.user_answer_embedding(user_answer))        
        
        
        #Add embeddings Encoder
        src = self.dropout_5(self.encoder(src))  #Embedding
        src = torch.add(src, part_emb)
        src = torch.add(src, ts_emb)   #Last interaction days 
        src = self.pos_encoder(src)   #Pos embedding
        
        
        #Add embedding decoder
        trg = self.dropout_6(self.decoder(trg))
        trg = torch.add(trg, user_answer_emb)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask)
        

        output = self.fc_out(output)

        return output