import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer


data_dir = "../../data/"


batch_size = 512
seq_len = 100

#If True, the model would be trained on +70 Million rows, 20M otherwise
train_full = False

#Answer start token
correct_start_token = 2
user_answer_start_token = 4


#Transformer hyperparameter 
d_model = 128

decoder_layers = 2
encoder_layers = 2

dropout = 0.1 
ff_model = d_model*4

att_heads = d_model // 32


#Loading questions, and every question corresponding part
que_data = pd.read_csv( dir_data + "raw/questions.csv")
part_valus = que_data.part.values

unique_ques = len(que_data)


auxiliary = pd.read_pickle(data_dir + "processed/training.pickle")
val_group = pd.read_pickle(data_dir + "processed/validation.pickle")


class TextLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x, self.y, self.ts, self.user_answer = [], [], [], []
        
        for line in data:
            x, y, ts, user_answer = line
            
            self.x.append(x)
            self.y.append(y)      
            self.ts.append(ts)
            self.user_answer.append(user_answer)

    def __getitem__(self, index):
        return (torch.LongTensor(self.x[index]), 
                torch.LongTensor(self.y[index]), 
                torch.LongTensor(self.ts[index]),
               torch.LongTensor(self.user_answer[index]))

    def __len__(self):
        return len(self.x)



class TextCollate():
    
    def __call__(self, batch):
        
        x_padded = torch.LongTensor(seq_len, len(batch))
        y_padded = torch.LongTensor(seq_len, len(batch))        
        ts_padded = torch.LongTensor(seq_len, len(batch))     
        user_answer_padded = torch.LongTensor(seq_len, len(batch))

        for i in range(len(batch)):
            
            x = batch[i][0]
            x_padded[:x.size(0), i] = x
            
            y = batch[i][1]
            y_padded[:y.size(0), i] = y
            
            ts = batch[i][2]
            ts_padded[:y.size(0),i] = ts
            
            user_answer = batch[i][3]
            user_answer_padded[:y.size(0),i] = user_answer
            

        return x_padded, y_padded, ts_padded, user_answer_padded



pin_memory = True
num_workers = 2

trainset = TextLoader(auxiliary)
valset = TextLoader(val_group)

collate_fn = TextCollate()

train_loader = torch.utils.data.DataLoader(trainset, num_workers=num_workers, shuffle=True,
                          batch_size=batch_size, pin_memory=pin_memory,
                          drop_last=True, collate_fn=collate_fn)  #(seq_len, batch_size)

val_loader = torch.utils.data.DataLoader(valset, num_workers=num_workers, shuffle=False,
                        batch_size=batch_size, pin_memory=pin_memory,
                        drop_last=False, collate_fn=collate_fn)





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
part_valus = torch.LongTensor(part_valus).to(device)


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


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


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


que_emb_size = unique_ques

model = TransformerModel(que_emb_size, hidden=d_model,part_arr=part_valus, dec_layers=decoder_layers, enc_layers=encoder_layers, dropout=dropout, nheads=att_heads, ff_model=ff_model).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
print(model)



import torch.optim as optim


optimizer = NoamOpt(d_model, 1, 4000 ,optim.Adam(model.parameters(), lr=0))

#Since the objective of training is a binary classification
criterion = nn.BCEWithLogitsLoss()



#Add padding to decoder input
def add_shift(var, pad):
    
    var_pad = torch.ShortTensor(1, var.shape[1]).to(device)
    var_pad.fill_(pad)
    
    return torch.cat((var_pad, var))


def train(model, optimizer, criterion, iterator):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src, trg, ts, user_answer = batch
        src, trg, ts, user_answer = src.to(device), trg.to(device), ts.to(device), user_answer.to(device)

        
        trg = add_shift(trg, correct_start_token)
        user_answer = add_shift(user_answer, user_answer_start_token)        
        
        optimizer.optimizer.zero_grad()
        output = model(src, trg[:-1,:], ts, user_answer[:-1,:])
        
        loss = criterion(output.squeeze(), trg[1:,:].float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


from sklearn.metrics import roc_auc_score

def evaluate(model, criterion, iterator):

    model.eval()

    epoch_loss = 0
    acc = 0
    
    preds = []
    corr = []

    with torch.no_grad():    
        for i, batch in enumerate(iterator):


            src, trg, ts, user_answer = batch
            src, trg, ts, user_answer = src.to(device), trg.to(device), ts.to(device), user_answer.to(device)


            trg = add_shift(trg, correct_start_token)
            user_answer = add_shift(user_answer, user_answer_start_token)        

            optimizer.optimizer.zero_grad()
            output = model(src, trg[:-1,:], ts, user_answer[:-1,:])
            loss = criterion(output.squeeze(), trg[1:,:].float())
            
            preds.extend(F.sigmoid(output).squeeze().reshape(-1).detach().cpu().numpy().tolist())
            corr.extend(trg[1:,:].reshape(-1).detach().cpu().numpy().tolist())
            
            nb_correct = F.sigmoid(output).squeeze().transpose(0, 1).round().reshape(-1) == trg[1:,:].float().transpose(0, 1).reshape(-1)
            accuracy = nb_correct.sum()/float(output.squeeze().transpose(0, 1).round().reshape(-1).shape[0])
            
            
            epoch_loss += loss.item()
            
            acc += accuracy.item()
            
    
            
    return (epoch_loss / len(iterator), acc/len(iterator), roc_auc_score(np.array(corr),np.array(preds)))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 65

best_roc = 0

for epoch in range(N_EPOCHS):
    print(f'Epoch: {epoch+1:02} ({best_roc})')

    start_time = time.time()

    train_loss = train(model, optimizer, criterion, train_loader)
    valid_loss, acc, roc = evaluate(model, criterion, val_loader)

    epoch_mins, epoch_secs = epoch_time(start_time, time.time())

    if roc > best_roc:
        best_roc = roc
        torch.save(model.state_dict(), '../../models/model_best.torch')

    print(f'Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss: {train_loss:.3f}')
    print(f'Val   Loss: {valid_loss:.3f}    Acc {acc:.3f}   ROC {roc:.3f}')
    
print(best_roc)