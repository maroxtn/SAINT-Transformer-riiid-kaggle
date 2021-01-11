#Note my code
#Credits goes to: https://www.kaggle.com/its7171/cv-strategy
#Creates the validation split from raw data and store them in data/interim
import pandas as pd
import random
import gc

import yaml

random.seed(1)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

train = pd.read_csv('data/raw/train.csv',
                   dtype={'row_id': 'int64',
                          'timestamp': 'int64',
                          'user_id': 'int32',
                          'content_id': 'int16',
                          'content_type_id': 'int8',
                          'task_container_id': 'int16',
                          'user_answer': 'int8',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'float32',
                          'prior_question_had_explanation': 'boolean'}
                   )

    
max_timestamp_u = train[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
max_timestamp_u.columns = ['user_id', 'max_time_stamp']
MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()



def rand_time(max_time_stamp):
    interval = MAX_TIME_STAMP - max_time_stamp
    rand_time_stamp = random.randint(0,interval)
    return rand_time_stamp

max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.apply(rand_time)
train = train.merge(max_timestamp_u, on='user_id', how='left')
train['viretual_time_stamp'] = train.timestamp + train['rand_time_stamp']


del train['max_time_stamp']
del train['rand_time_stamp']
del max_timestamp_u
gc.collect()


train = train.sort_values(['viretual_time_stamp', 'row_id']).reset_index(drop=True)

val_size = int(train.shape[0] * 0.025)  #Would be roughly 2.5M for full data

valid = train[-val_size:]
train = train[:-val_size]

valid.to_pickle(f'data/interim/cv_valid.pickle')
train.to_pickle(f'data/interim/cv_train.pickle')