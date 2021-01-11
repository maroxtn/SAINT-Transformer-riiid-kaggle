# SAINT Transformer model

This is my solution for the Riiid knowledge tracing competition, which was ensembled with an LGBM model to give me the 39th rank.

For the code to be run properly, the dataset should be in `/data/raw` then:

- `src/data/validation_split.py` should be ran so we have both validation and training data
- `src/data/preprocess.py` will create the training and validation data preprocessed, and will create a file with user's history, all pickled to data/preprocessed
- `src/models/train.py` will train the model, and save it in `models/`

**Note** that running all the transformations on the raw data would require at least 32GB of ram, training will take 15 minutes per epoch for the entire data. 

-----

I have created an interactive notebook with more thorough explanation on kaggle in which you can train and test the inference on the dataset, you can take a look at it here: https://www.kaggle.com/abdessalemboukil/saint-training-inference-guide-39th-solution


SAINT+ paper: https://arxiv.org/pdf/2010.12042.pdf