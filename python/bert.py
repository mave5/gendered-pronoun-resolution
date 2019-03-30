import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import zipfile
import sys
import time
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import log_loss
import time

# Download weights and cofiguration file for the model
# $wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
#path2zip="/home/mra/Downloads/uncased_L-12_H-768_A-12.zip"
#with zipfile.ZipFile(path2zip,"r") as zip_ref:
#    zip_ref.extractall()


from utils import modeling
from utils import extract_features
from utils import tokenization
import tensorflow as tf
from utils.utils_bert import *

#%%
DATA_ROOT = '../data/'
GAP_DATA_FOLDER = os.path.join(DATA_ROOT, 'gap-coreference')
SUB_DATA_FOLDER = os.path.join(DATA_ROOT, 'gendered-pronoun-resolution')
FAST_TEXT_DATA_FOLDER = os.path.join(DATA_ROOT, 'fasttext-crawl-300d-2m')

path2testJson=os.path.join(DATA_ROOT,"contextual_embeddings_gap_test.json")
path2validJson=os.path.join(DATA_ROOT,"contextual_embeddings_gap_validation.json")
path2devJson=os.path.join(DATA_ROOT,"contextual_embeddings_gap_development.json")

#%% 

# Get contextual embeddings in json files.
test_data = pd.read_csv(GAP_DATA_FOLDER+"/gap-test.tsv", sep = '\t')
if not os.path.exists(path2testJson): 
    test_emb = run_bert(test_data)
    test_emb.to_json(path2testJson, orient = 'columns')
else:
    test_emb = pd.read_json(path2testJson, lines = True)
    print(test_emb.columns)

validation_data = pd.read_csv(GAP_DATA_FOLDER+"/gap-validation.tsv", sep = '\t')
if not os.path.exists(path2validJson): 
    validation_emb = run_bert(validation_data)
    validation_emb.to_json(path2validJson, orient = 'columns')
else:
    validation_emb = pd.read_json(path2validJson, lines = True)
    print(validation_emb.columns)

development_data = pd.read_csv(GAP_DATA_FOLDER+"/gap-development.tsv", sep = '\t')
if not os.path.exists(path2devJson): 
    development_emb = run_bert(development_data)
    development_emb.to_json(path2devJson, orient = 'columns')
else:
    development_emb = pd.read_json(path2devJson, lines = True)
    print(development_emb.columns)

#%%

# Read development embeddigns from json file - this is the output of Bert
development = pd.read_json(path2devJson)
X_development, Y_development = parse_json(development)

validation = pd.read_json(path2validJson)
X_validation, Y_validation = parse_json(validation)

test = pd.read_json(path2testJson)
X_test, Y_test = parse_json(test)

#%%

# There may be a few NaN values, where the offset of a target word is greater than the max_seq_length of BERT.
# They are very few, so I'm just dropping the rows.
remove_test = [row for row in range(len(X_test)) if np.sum(np.isnan(X_test[row]))]
X_test = np.delete(X_test, remove_test, 0)
Y_test = np.delete(Y_test, remove_test, 0)

remove_validation = [row for row in range(len(X_validation)) if np.sum(np.isnan(X_validation[row]))]
X_validation = np.delete(X_validation, remove_validation, 0)
Y_validation = np.delete(Y_validation, remove_validation, 0)

# We want predictions for all development rows. So instead of removing rows, make them 0
remove_development = [row for row in range(len(X_development)) if np.sum(np.isnan(X_development[row]))]
X_development[remove_development] = np.zeros(3*768)

#%%

# Will train on data from the gap-test and gap-validation files, in total 2454 rows
X_train = np.concatenate((X_test, X_validation), axis = 0)
Y_train = np.concatenate((Y_test, Y_validation), axis = 0)

# Will predict probabilities for data from the gap-development file; initializing the predictions
prediction = np.zeros((len(X_development),3)) # testing predictions
#%%
learning_rate = 0.001
n_fold = 5
batch_size = 32
epochs = 1000
patience = 100
# n_test = 100

params_model={
        "dense_layer_sizes": [100],
        "dropout_rate": 0.6,
        "input_shape": [X_train.shape[1]],
        "lambd": 0.01, # L2 regularization  
        "bnEnable": True,
        }

# Training and cross-validation
folds = KFold(n_splits=n_fold, shuffle=True, random_state=3)
scores = []
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
	# split training and validation data
    print('Fold', fold_n, 'started at', time.ctime())
    X_tr, X_val = X_train[train_index], X_train[valid_index]
    Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]
    
    # Define the model, re-initializing for each fold
    classif_model = build_mlp_model(params_model)
    classif_model.summary()
    classif_model.compile(optimizer = optimizers.Adam(lr = learning_rate), loss = "categorical_crossentropy")
    callbacks = [kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights = True)]
	# train the model
    classif_model.fit(x = X_tr, y = Y_tr, epochs = epochs, batch_size = batch_size, callbacks = callbacks, validation_data = (X_val, Y_val), verbose = 0)
    
    # make predictions on validation and test data
    pred_valid = classif_model.predict(x = X_val, verbose = 0)
    pred = classif_model.predict(x = X_development, verbose = 0)
    
    # oof[valid_index] = pred_valid.reshape(-1,)
    score_val=log_loss(Y_val, pred_valid)
    scores.append(score_val)
    print("Fold: %s, score: %s " %(fold_n+1,score_val))
    print("-"*50)
    prediction += pred
prediction /= n_fold

# Print CV scores, as well as score on the test data
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
print(scores)
print("Test score:", log_loss(Y_development,prediction))

#%%
# Write the prediction to file for submission
sub_df_path = os.path.join(SUB_DATA_FOLDER, 'sample_submission_stage_1.csv')
submissionDF = pd.read_csv(sub_df_path, index_col = "ID")
submissionDF["A"] = prediction[:,0]
submissionDF["B"] = prediction[:,1]
submissionDF["NEITHER"] = prediction[:,2]

import datetime
submissionFolder="./submissions/"
if not os.path.exists(submissionFolder):
    os.mkdir(submissionFolder)
    print(submissionFolder+" created!")

info="bert5folds"
now = datetime.datetime.now()
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
path2submission = os.path.join(submissionFolder, 'submission_' + suffix + '.csv')
print(path2submission)
submissionDF.to_csv(path2submission)

