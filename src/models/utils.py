import torch
import yaml
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)



correct_start_token = config["correct_start_token"]
user_answer_start_token = config["user_answer_start_token"]



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
