import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

# from spacy.lang.en import English
# from spacy.pipeline import DependencyParser
import spacy
from nltk import Tree
from keras import models
from sklearn.metrics import accuracy_score
from utils.utils_coref import *
import matplotlib.pyplot as plt
# from IPython.display import SVG

np.random.seed(2019)
histories = list()

nlp = spacy.load('en_core_web_sm')

# In[2]: pre-sets

DATA_ROOT = '../data/'
GAP_DATA_FOLDER = os.path.join(DATA_ROOT, 'gap-coreference')
SUB_DATA_FOLDER = os.path.join(DATA_ROOT, 'gendered-pronoun-resolution')
FAST_TEXT_DATA_FOLDER = os.path.join(DATA_ROOT, 'fasttext-crawl-300d-2m')

num_embed_features = 11
embed_dim = 384
num_pos_features = 45


# In[3]: loading data

test_df_path = os.path.join(GAP_DATA_FOLDER, 'gap-development.tsv')
train_df_path = os.path.join(GAP_DATA_FOLDER, 'gap-test.tsv')
dev_df_path = os.path.join(GAP_DATA_FOLDER, 'gap-validation.tsv')

train_df = pd.read_csv(train_df_path, sep='\t')
test_df = pd.read_csv(test_df_path, sep='\t')
dev_df = pd.read_csv(dev_df_path, sep='\t')

# pd.options.display.max_colwidth = 1000
train_df.head()
test_df.head()
dev_df.head()

print("training data shape:",train_df.shape)
print("dev data shape:",dev_df.shape)
print("test data shape:",test_df.shape)
print("-"*50)

print ("data frame columns:",train_df.columns)
print("-"*50)
#%% Obtain features for a sample text
k1=np.random.randint(len(train_df))
text = train_df.Text[k1]
pronoun_offset=train_df["Pronoun-offset"][k1]

print("\nDependency parsing trees: ")
doc = nlp(text) 
[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

print("\nFeatures:")
print("-"*50)
for col in train_df.columns:
    print(col,"=> ",train_df[col][k1])

embeddingFeatures = extrac_embed_features_tokens(text, pronoun_offset)
features = pd.Series([str(feature) for feature in embeddingFeatures.values()], index=list(embeddingFeatures.keys()))
features.head()

#%% Obtain Embedding feature Matrix for Pronoun in Train, Dev, Test data

print("wait ...")
p_emb_train = create_embedding_features(train_df, 'Text', 'Pronoun-offset',num_embed_features,embed_dim)
print("embedding features training:", p_emb_train.shape)
p_emb_dev = create_embedding_features(dev_df, 'Text', 'Pronoun-offset',num_embed_features,embed_dim)
print("embedding features dev:", p_emb_dev.shape)
p_emb_test = create_embedding_features(test_df, 'Text', 'Pronoun-offset',num_embed_features,embed_dim)
print("embedding features test:", p_emb_test.shape)

#%% Obtain Embedding feature Matrix for candidate-A in Train, Dev, Test data
a_emb_train = create_embedding_features(train_df, 'Text', 'A-offset')
print(a_emb_train.shape)
a_emb_dev = create_embedding_features(dev_df, 'Text', 'A-offset')
print(a_emb_dev.shape)
a_emb_test = create_embedding_features(test_df, 'Text', 'A-offset')
print(a_emb_test.shape)

#%% Obtain Embedding feature Matrix for candidate-B in Train, Dev, Test data
b_emb_train = create_embedding_features(train_df, 'Text', 'B-offset')
print(b_emb_train.shape)
b_emb_dev = create_embedding_features(dev_df, 'Text', 'B-offset')
print(b_emb_dev.shape)
b_emb_test = create_embedding_features(test_df, 'Text', 'B-offset')
print(b_emb_test.shape)


#%% Obtain Distance Features: Pronoun to Candidate-A

pa_pos_train = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'A-offset',num_pos_features)
print(pa_pos_train.shape)
pa_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'A-offset',num_pos_features)
print(pa_pos_dev.shape)
pa_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'A-offset',num_pos_features)
print(pa_pos_test.shape)

#%% Obtain Distance Features: Pronoun to Candidate-B
pb_pos_train = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'B-offset',num_pos_features)
print(pb_pos_train.shape)
pb_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'B-offset',num_pos_features)
print(pb_pos_dev.shape)
pb_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'B-offset',num_pos_features)
print(pb_pos_test.shape)


#%% Obtain labels
y_train = getLabels(train_df) 
y_dev = getLabels(dev_df)
y_test = getLabels(test_df)
print(y_train.shape)
print(y_dev.shape)
print(y_test.shape)

#%% Concatenate features 
X_train = [p_emb_train, a_emb_train, b_emb_train, pa_pos_train, pb_pos_train]
X_dev = [p_emb_dev, a_emb_dev, b_emb_dev, pa_pos_dev, pb_pos_dev]
X_test = [p_emb_test, a_emb_test, b_emb_test, pa_pos_test, pb_pos_test]
print(len(X_train))
print(len(X_dev))
print(len(X_test))


#%% Model definition and training

params_mlp={
        "num_feature_channels1": 3, # channels of embedding features
        "num_feature_channels2": 2, # channels of pos features
        "num_features1": num_embed_features, 
        "num_features2": num_pos_features,
        "feature_dim1": embed_dim,
        "output_dim": 3,
        "model_dim": 10, # number of neurons in the first hidden layer
        "mlp_depth": 1, # number of hidden layers: Depth
        "mlp_dim": 60,  # number of neurons in other hidden layers
        "drop_out": 0.5,
        "return_customized_layers": True,
        "loss": "sparse_categorical_crossentropy", # accepts integer as label
        "optimizer": "Nadam",
        "metrics": ["sparse_categorical_accuracy"],
        }

model, co_mlp = build_mlp_model(params_mlp)
model.summary()

params_train_test={
        "model_name": "best_mlp_model",
        "X_train": X_train,
        "y_train": y_train,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "patience": 20,
        "epochs": 100,
        "batch_size": 16,
        }

minLoss,path2model=train_test(model,params_train_test)
print("best loss: %.3f" %minLoss)
histories.append(minLoss)
del model
gc.collect()

model=models.load_model(path2model)

for xx,yy in zip([X_train,X_dev,X_test],[y_train,y_dev,y_test]):
    lossAccuracy=model.evaluate(xx,yy,verbose=0)
    print("loss, accuracy: ", lossAccuracy)


#%%

params_cnn={
        "num_feature_channels1": 3, # channels of embedding features
        "num_feature_channels2": 2, # channels of pos features
        "num_features1": num_embed_features, 
        "num_features2": num_pos_features,
        "feature_dim1": embed_dim,
        "output_dim": 3,
        "model_dim": 10, # number of neurons in the first hidden layer
        "mlp_depth": 1, # number of hidden layers: Depth
        "mlp_dim": 60,  # number of neurons in other hidden layers
        "drop_out": 0.5,
        "return_customized_layers": True,
        "loss": "sparse_categorical_crossentropy", # accepts integer as label
        "optimizer": "Nadam",
        "metrics": ["sparse_categorical_accuracy"],
        "filter_sizes": [3, 5],
        "num_filters": [10] * 2, # [model_dim] * len(filter_sizes)
        "pooling": "max",
        "padding": "valid",
        }

model, co_mccnn = build_multi_channel_cnn_model(params_cnn)
model.summary()

params_train_test={
        "model_name": "best_mc_cnn_model",
        "X_train": X_train,
        "y_train": y_train,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "patience": 20,
        "epochs": 100,
        "batch_size": 16,
        }


minLoss,path2model=train_test(model,params_train_test)

print("best loss: %.3f" %minLoss)
histories.append(minLoss)
del model
gc.collect()

model=models.load_model(path2model)

for xx,yy in zip([X_train,X_dev,X_test],[y_train,y_dev,y_test]):
    lossAccuracy=model.evaluate(xx,yy,verbose=0)
    print("loss, accuracy: ", lossAccuracy)


# In[29]:

params_coattention_cnn={
        "num_feature_channels1": 3, # channels of embedding features
        "num_feature_channels2": 2, # channels of pos features
        "num_features1": num_embed_features, 
        "num_features2": num_pos_features,
        "feature_dim1": embed_dim,
        "output_dim": 3,
        "model_dim": 10, # number of neurons in the first hidden layer
        "mlp_depth": 1, # number of hidden layers: Depth
        "mlp_dim": 5,  # number of neurons in other hidden layers
        "drop_out": 0.5,
        "return_customized_layers": True,
        "loss": "sparse_categorical_crossentropy", # accepts integer as label
        "optimizer": "Nadam",
        "metrics": ["sparse_categorical_accuracy"],
        "filter_sizes": [1],
        "num_filters": [20] * 1, # [model_dim] * len(filter_sizes)
        "pooling": "max",
        "padding": "valid",
        "atten_dim": 10,
        }

model, co_cacnn = build_inter_coattention_cnn_model(params_coattention_cnn)
model.summary()

params_train_test={
        "model_name": "best_coatt_cnn_model",
        "X_train": X_train,
        "y_train": y_train,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "patience": 20,
        "epochs": 100,
        "batch_size": 16,
        }

minLoss,path2model=train_test(model,params_train_test)
print("best loss: %.3f" %minLoss)
histories.append(minLoss)
del model
gc.collect()

model=models.load_model(path2model,co_cacnn)

for xx,yy in zip([X_train,X_dev,X_test],[y_train,y_dev,y_test]):
    lossAccuracy=model.evaluate(xx,yy,verbose=0)
    print("loss, accuracy: ", lossAccuracy)


# In[31]:

params_intra_coattention_cnn={
        "num_feature_channels1": 3, # channels of embedding features
        "num_feature_channels2": 2, # channels of pos features
        "num_features1": num_embed_features, 
        "num_features2": num_pos_features,
        "feature_dim1": embed_dim,
        "output_dim": 3,
        "model_dim": 10, # number of neurons in the first hidden layer
        "mlp_depth": 1, # number of hidden layers: Depth
        "mlp_dim": 5,  # number of neurons in other hidden layers
        "drop_out": 0.5,
        "return_customized_layers": True,
        "loss": "sparse_categorical_crossentropy", # accepts integer as label
        "optimizer": "Nadam",
        "metrics": ["sparse_categorical_accuracy"],
        "filter_sizes": [1],
        "num_filters": [20] * 1, # [model_dim] * len(filter_sizes)
        "pooling": "max",
        "padding": "valid",
        "atten_dim": 10,
        }


model, intra_co_cacnn = build_intra_coattention_cnn_model(params_intra_coattention_cnn)
model.summary()


params_train_test={
        "model_name": "best_intra_coatt_cnn_model",
        "X_train": X_train,
        "y_train": y_train,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "patience": 20,
        "epochs": 100,
        "batch_size": 16,
        }

minLoss,path2model=train_test(model,params_train_test)
print("best loss: %.3f" %minLoss)
histories.append(minLoss)
del model
gc.collect()

model=models.load_model(path2model,intra_co_cacnn)

for xx,yy in zip([X_train,X_dev,X_test],[y_train,y_dev,y_test]):
    lossAccuracy=model.evaluate(xx,yy,verbose=0)
    print("loss, accuracy: ", lossAccuracy)


# In[33]:
weightFolder="./weights"
model_paths = [
    os.path.join(weightFolder,"best_mlp_model.hdf5"),
    os.path.join(weightFolder,"best_mc_cnn_model.hdf5"),
    os.path.join(weightFolder,"best_coatt_cnn_model.hdf5"),
    os.path.join(weightFolder,"best_intra_coatt_cnn_model.hdf5")
]

# custom layers
customLayers =[co_mlp, co_mccnn, co_cacnn, intra_co_cacnn]

numOfModels=4
y_preds=np.zeros((len(y_test),3))
for p2m,cl in zip(model_paths,customLayers):
    print("loading model: ", p2m)
    model = models.load_model(p2m, cl)
    y_preds+= model.predict(X_test, batch_size = 1024, verbose = 0)/numOfModels

#accuracy_score(y_test,np.argmax(y_preds,axis=1))
sub_df_path = os.path.join(SUB_DATA_FOLDER, 'sample_submission_stage_1.csv')
submissionDF = pd.read_csv(sub_df_path)
submissionDF.loc[:, 'A'] = pd.Series(y_preds[:, 0])
submissionDF.loc[:, 'B'] = pd.Series(y_preds[:, 1])
submissionDF.loc[:, 'NEITHER'] = pd.Series(y_preds[:, 2])

submissionDF.head()


#%% create Submission

import datetime
submissionFolder="./submissions/"
if not os.path.exists(submissionFolder):
    os.mkdir(submissionFolder)
    print(submissionFolder+" created!")

info="ensemble"
now = datetime.datetime.now()
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
path2submission = os.path.join(submissionFolder, 'submission_' + suffix + '.csv')
print(path2submission)
submissionDF.to_csv(path2submission, index=False)

