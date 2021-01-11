#Preprocess the data, created training and validation processed data, 
# and prepare files necessary for inference

import sys
import numpy as np

import os
import glob
import time
from os import listdir
from typing import Dict
import datatable as dt

import sklearn.preprocessing as preprocessing
from sklearn.metrics import roc_auc_score

import pandas as pd

from sys import getsizeof

#supress warnings
import warnings
warnings.filterwarnings("ignore")

import time
import gc
import math 



data_dir = "../../data/"


seq_len = 100

#If True, the model would be trained on +70 Million rows, 20M otherwise
train_full = False


#Load the validation / training splitted data created by Tito. The validation data occur after the training data.
train_data = pd.read_pickle(dir_data + "interim/cv_train.pickle")
validation = pd.read_pickle(dir_data + "interim/cv_valid.pickle")


#Remove uneeded rows, and drop lectures for the training data
del train_data["prior_question_had_explanation"]
del train_data["prior_question_elapsed_time"]
del train_data["max_time_stamp"]
del train_data["rand_time_stamp"]
del train_data["viretual_time_stamp"]

del validation["prior_question_had_explanation"]
del validation["prior_question_elapsed_time"]
del validation["max_time_stamp"]
del validation["rand_time_stamp"]
del validation["viretual_time_stamp"]

train_data = train_data[train_data.content_type_id == False]
del train_data["content_type_id"]
del train_data["row_id"]



last_timestamp = train_data.groupby("user_id")[["timestamp","user_id"]].tail(1).set_index("user_id", drop=True)["timestamp"].to_dict()
train_data.reset_index(drop=True, inplace=True) #Resetting the index frees memory



train_data["timestamp"] = train_data.groupby("user_id")["timestamp"].diff().fillna(0)/1000
train_data["timestamp"] = train_data.timestamp.astype("int32")



boundaries = [120,600,1800,3600,10800,43200,86400,259200,604800]
x = train_data.timestamp.copy()

for i, boundary in enumerate(boundaries):
    
    if i == 0:
        start = 60
    else:
        start = boundaries[i-1]
        
    end = boundary
    
    train_data.loc[(x >= start) & (x < end), "timestamp"] = i+60
    
train_data.loc[x >= end, "timestamp"] = i+60+1

del x
train_data["timestamp"] =train_data["timestamp"].astype("int8")
gc.collect()



group = train_data[['user_id', 'content_id', 'answered_correctly', 'timestamp',"user_answer"]].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values, r['timestamp'].values,r['user_answer'].values))


#Creating the validation data
user_counts = group.apply(lambda x: len(x[0])).sort_values(ascending=False)
user_counts = user_counts[(user_counts >= seq_len)]

accepted_ids = user_counts.index
val_group = group.loc[accepted_ids]


def f(x):
    return (x[0][:seq_len], x[1][:seq_len], x[2][:seq_len], x[3][:seq_len])

val_group = val_group.apply(f).sample(frac=0.1)
group = group.drop(index=val_group.index)




#Creating sequences of 100 of the all interactions less than 1000
user_counts = group.apply(lambda x: len(x[0])).sort_values(ascending=False)

if train_full:  #Train with subset of data or not
    user_counts = user_counts[(user_counts >= seq_len)]
else:
    user_counts = user_counts[((user_counts >= seq_len) & (user_counts <= 1000))]

accepted_ids = user_counts.index
group = group.loc[accepted_ids]

group.index = group.index.astype("str")



from tqdm import tqdm

auxiliary = []
k = 0
    
for line in tqdm(group):
    
    src, trg, ts, user_answer = line
    chunk_len = seq_len
    i = 0
    
    split_size = src.shape[0] - src.shape[0]%chunk_len
    n_splits = split_size/chunk_len
    
    lst = list(zip(np.split(src[:split_size],n_splits), 
                   np.split(trg[:split_size], n_splits), 
                   np.split(ts[:split_size], n_splits), 
                   np.split(user_answer[:split_size], n_splits),
                  ))
    
    auxiliary.extend(lst) 


#Training and validation data
auxiliary.to_pickle(data_dir + "processed/training.pickle")
val_group.to_pickle(data_dir + "processed/validation.pickle")



#Generate the data used for inference
group = train_data[['user_id', 'content_id', 'answered_correctly', 'timestamp',"user_answer"]].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values, r['timestamp'].values,r['user_answer'].values))

group.to_pickle(data_dir + "processed/inference_group")

with open(data_dir + 'inference_last_timestamp.pickle', 'wb') as handle:
    pickle.dump(last_timestamp, handle, protocol=pickle.HIGHEST_PROTOCOL)
