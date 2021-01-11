# SAINT Transformer model

This is my solution for the Riiid knowledge tracing competition, which was ensembled with an LGBM model to give me the 39th rank. 

To run the code you should: 

- Download the dataset using either the kaggle api or manually to `data/raw/` , questions.csv and train.csv only are going to be used.
- Configure the model hyper-parameter, and general config in `config.yaml`
- Run: `python src/data/validation_split.py` to create the validation data
- Run: `python src/data/preprocess.py` to preprocess the data, and make training and validation into the right format
- Run: `src/modelds/train.py` to train the model.

Feel free to check the source, I tried to make it as readable as possible. After finishing the training, you can play with the inference code, with the kaggle time series API emulator in notebooks directory.

-----

**Note** that running all the transformations on the raw data would require at least 32GB of ram, training will take 15 minutes per epoch for the entire data. If all data is used, training would use 70M + sequences.

-----

I have created an interactive notebook with more thorough explanation on kaggle in which you can train and test the inference on the dataset, you can take a look at it here: https://www.kaggle.com/abdessalemboukil/saint-training-inference-guide-39th-solution


SAINT+ paper: https://arxiv.org/pdf/2010.12042.pdf