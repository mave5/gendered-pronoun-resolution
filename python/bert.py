import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import zipfile
import sys
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
    test_emb.head()

validation_data = pd.read_csv(GAP_DATA_FOLDER+"/gap-validation.tsv", sep = '\t')
if not os.path.exists(path2validJson): 
    validation_emb = run_bert(validation_data)
    validation_emb.to_json(path2validJson, orient = 'columns')
else:
    validation_emb = pd.read_json(path2validJson, lines = True)
    validation_emb.head()

development_data = pd.read_csv(GAP_DATA_FOLDER+"/gap-development.tsv", sep = '\t')
if not os.path.exists(path2devJson): 
    development_emb = run_bert(development_data)
    development_emb.to_json(path2devJson, orient = 'columns')
else:
    development_emb = pd.read_json(path2devJson, lines = True)
    development_emb.head()


