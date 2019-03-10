from nltk import Tree
import spacy
import numpy as np
import os
nlp = spacy.load('en_core_web_sm')

# from keras import backend
from keras import layers
from keras import models
from keras import initializers, regularizers, constraints, activations
from keras.engine import Layer
import keras.backend as K
from keras.layers import merge
from keras import callbacks as kc
from keras import optimizers as ko
# from keras import initializers, regularizers, constraints


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def binarySearch(list_, target_):
    lo, hi = 0, len(list_) -1
    
    while lo < hi:
        mid = lo + int((hi - lo) / 2)
        
        if target_ < list_[mid]:
            hi = mid
        elif target_ > list_[mid]:
            lo = mid + 1
        else:
            return mid + 1
    return lo


def _get_preceding_words(tokens, offset, k):
    start = offset - k
    
    precedings = [None] * max(0, 0-start)
    start = max(0, start)
    precedings += tokens[start: offset]
    
    return precedings

def _get_following_words(tokens, offset, k):
    end = offset + k
    
    followings = [None] * max(0, end - len(tokens))
    end = min(len(tokens), end)
    followings += tokens[offset+1: end+1] # added +1 to offset to select the following word!
    
    return followings
        

def extrac_embed_features_tokens(text, char_offset):
    # tokenize
    doc = nlp(text)
    
    # finding char offset for every token in the text
    token_offsets = [token.idx for token in doc] 
    
    # Finding index of mention in the tokenized text
    mention_index = binarySearch(token_offsets, char_offset) - 1 
    assert mention_index >= 0
    
    # mention_word
    mention = doc[mention_index]
    
    # Finding which sentence contains the mention
    lengthOfSentences = [len(sent) for sent in doc.sents]
    
    pre_len = 0
    sentenceEndIndex=[]
    for len_ in lengthOfSentences:
        pre_len += len_
        sentenceEndIndex.append(pre_len)
    sent_index = binarySearch(sentenceEndIndex, mention_index)
    
    # mention sentence
    mention_sentence = list(doc.sents)[sent_index]
    
    # dependency parent
    # check out: http://www.nltk.org/book/ch08.html#fig-depgraph0
    head = mention.head
    
    # last word and first word
    first_word, last_word = mention_sentence[0], mention_sentence[-2]
    
    # two preceding words and two following words
    tokens = list(doc)
    precedings2 = _get_preceding_words(tokens, mention_index, 2)
    followings2 = _get_following_words(tokens, mention_index, 2)
    
    # five preceding words and five following words
    precedings5 = _get_preceding_words(tokens, mention_index, 5)
    followings5 = _get_following_words(tokens, mention_index, 5)
    
    # sentence words
    mention_sent_tokens = [token for token in mention_sentence]
    
    # Paper"mproving Coreference Resolution by Learning Entity-Level DistributedRepresentations", Kevin Clark, et al    
    embeddingFeatures={
            "mention": mention,
            "parent": head,
            "first_word": first_word,
            "last_word": last_word,
            "precedings2": precedings2,
            "followings2": followings2,
            "precedings5": precedings5,
            "followings5": followings5,
            "sent_tokens": mention_sent_tokens,
            }
    return embeddingFeatures


def create_embedding_features(df, text_column, offset_column,num_embed_features=11,embed_dim=384):
    text_offset_list = df[[text_column, offset_column]].values.tolist()
    num_features = num_embed_features
    
    # initialize feature matrix
    embed_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features, embed_dim))
    
    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        
        # extract features
        embFea= extrac_embed_features_tokens(text_offset[0], text_offset[1])
         
        mention=embFea["mention"]
        parent=embFea["parent"]
        first_word=embFea["first_word"]
        last_word=embFea["last_word"]
        precedings2=embFea["precedings2"]
        followings2=embFea["followings2"]
        precedings5=embFea["precedings5"]
        followings5=embFea["followings5"]
        sent_tokens=embFea["sent_tokens"]
        
        feature_index = 0
        embed_feature_matrix[text_offset_index, feature_index, :] = mention.vector
        
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = parent.vector
        
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = first_word.vector
        
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = last_word.vector
        
        feature_index += 1
        procedings2_vector=np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings2])
        embed_feature_matrix[text_offset_index, feature_index:feature_index+2, :] = procedings2_vector
        
        feature_index += len(precedings2)
        followings2_vector=np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in followings2])
        embed_feature_matrix[text_offset_index, feature_index:feature_index+2, :] = followings2_vector
        
        # Averaged  word  embeddings of the five preceding words
        feature_index += len(followings2)
        precedings5_vecotr=np.mean(np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings5]), axis=0)
        embed_feature_matrix[text_offset_index, feature_index, :] = precedings5_vecotr
        
        # Averaged  word  embed-dings of the five following words
        feature_index += 1
        followings5_vector=np.mean(np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in followings5]), axis=0)
        embed_feature_matrix[text_offset_index, feature_index, :] = followings5_vector
        
        # all words in the mention's sentence
        feature_index += 1
        sent_tokens_vector=np.mean(np.asarray([token.vector for token in sent_tokens]), axis=0) if len(sent_tokens) > 0 else np.zeros(embed_dim)
        embed_feature_matrix[text_offset_index, feature_index, :] = sent_tokens_vector
        feature_index += 1
    
    return embed_feature_matrix


def binarySearch2(list_, target_):
    lo, hi = 0, len(list_) -1
    
    while lo < hi:
        mid = lo + int((hi - lo) / 2)
        
        if target_ < list_[mid]:
            hi = mid
        elif target_ > list_[mid]:
            lo = mid + 1
        else:
            return mid
    return lo

def oneHotEncodingDistance(dist, buckets):
    idx = binarySearch2(buckets, dist)
    oh = np.zeros(shape=(len(buckets),), dtype=np.float32)
    oh[idx] = 1
    
    return oh


def extrac_positional_features(text, char1_offset, char2_offset):
    doc = nlp(text)
    # max_len = 64
    
    # finding char offset for every token in the text
    token_offsets = [token.idx for token in doc]
    
    # index of each mention in tokenized text
    mention1_index = binarySearch(token_offsets, char1_offset) - 1
    mention2_index = binarySearch(token_offsets, char2_offset) - 1
    
    # length of sentences
    lengthOfSentences = [len(sent) for sent in doc.sents]
    
    pre_len = 0
    sentenceEndIndex=[]
    for len_ in lengthOfSentences:
        pre_len += len_
        sentenceEndIndex.append(pre_len)
    sent1_index = binarySearch(sentenceEndIndex, mention1_index)
    sent2_index = binarySearch(sentenceEndIndex, mention2_index)
    
    sent1 = list(doc.sents)[sent1_index]
    sent2 = list(doc.sents)[sent2_index]
    
    # buckets
    bucket_dist = [1, 2, 3, 4, 5, 8, 16, 32, 64]
    
    # relative distance
    distM1M2 = mention2_index - mention1_index
    dist_oh = oneHotEncodingDistance(distM1M2, bucket_dist)
    
    # buckets
    bucket_pos = [0, 1, 2, 3, 4, 5, 8, 16, 32]
    
    # absolute position in the sentence
    sent_pos1 = mention1_index + 1
    if sent1_index > 0:
        sent_pos1 = mention1_index - sentenceEndIndex[sent1_index-1]
    sent_pos_oh1 = oneHotEncodingDistance(sent_pos1, bucket_pos)
    sent_pos_inv1 = len(sent1) - sent_pos1
    assert sent_pos_inv1 >= 0
    sent_pos_inv_oh1 = oneHotEncodingDistance(sent_pos_inv1, bucket_pos)
    
    sent_pos2 = mention2_index + 1
    if sent2_index > 0:
        sent_pos2 = mention2_index - sentenceEndIndex[sent2_index-1]
    sent_pos_oh2 = oneHotEncodingDistance(sent_pos2, bucket_pos)
    sent_pos_inv2 = len(sent2) - sent_pos2
    if sent_pos_inv2 < 0:
        print(sent_pos_inv2)
        print(len(sent2))
        print(sent_pos2)
        raise ValueError
    sent_pos_inv_oh2 = oneHotEncodingDistance(sent_pos_inv2, bucket_pos)
    
    # sent_pos_ratio1 = sent_pos1 / len(sent1)
    # sent_pos_ratio2 = sent_pos2 / len(sent2)
    
    # return dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2
    posFeatures={
            "dist_oh": dist_oh,
            "sent_pos_oh1": sent_pos_oh1,
            "sent_pos_oh2": sent_pos_oh2,
            "sent_pos_inv_oh1": sent_pos_inv_oh1,
            "sent_pos_inv_oh2": sent_pos_inv_oh2,
            }
    return posFeatures

def create_dist_features(df, text_column, pronoun_offset_column, name_offset_column,num_pos_features=45):
    text_offset_list = df[[text_column, pronoun_offset_column, name_offset_column]].values.tolist()
    num_features = num_pos_features
    
    # initialize matrix
    pos_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features))
    
    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        posFeatures= extrac_positional_features(text_offset[0], text_offset[1], text_offset[2])
        dist_oh=posFeatures["dist_oh"]
        sent_pos_oh1=posFeatures["sent_pos_oh1"]
        sent_pos_oh2=posFeatures["sent_pos_oh2"]
        sent_pos_inv_oh1=posFeatures["sent_pos_inv_oh1"]
        sent_pos_inv_oh2=posFeatures["sent_pos_inv_oh2"]
        
        feature_index = 0
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(dist_oh)] = np.asarray(dist_oh)
        
        feature_index += len(dist_oh)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh1)] = np.asarray(sent_pos_oh1)
        
        feature_index += len(sent_pos_oh1)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh2)] = np.asarray(sent_pos_oh2)
        
        feature_index += len(sent_pos_oh2)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh1)] = np.asarray(sent_pos_inv_oh1)
        
        feature_index += len(sent_pos_inv_oh1)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh2)] = np.asarray(sent_pos_inv_oh2)
        feature_index += len(sent_pos_inv_oh2)
    
    return pos_feature_matrix


def _row_to_y(row):
    if row.loc['A-coref']:
        return 0
    if row.loc['B-coref']:
        return 1
    return 2

def getLabels(df):
    labels=df.apply(_row_to_y, axis=1)
    return labels


def build_mlp_model(params):
    """
    Create A Multi-Layer Perceptron Model.
    
    inputs: 
        embeddings: [batch, num_embed_feature, embed_dims] * 3 ## pronoun, A, B
        positional_features: [batch, num_pos_feature] * 2 ## pronoun-A, pronoun-B
        
    outputs: 
        [batch, num_classes] # in our case there should be 3 output classes: A, B, None
        
    :param output_dim: the output dimension size
    :param model_dim: rrn dimension size
    :param mlp_dim: the dimension size of fully connected layer
    :param mlp_depth: the depth of fully connected layers
    :param drop_out: dropout rate of fully connected layers
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
    :return: keras model
    """   
    num_feature_channels1=params["num_feature_channels1"] # number of channels in embedding features
    num_feature_channels2=params["num_feature_channels2"] # number of channels in positional features
    num_features1=params["num_features1"] # number of embeding features
    num_features2=params["num_features2"] # number of positional features
    feature_dim1=params["feature_dim1"] # embeding dimensions
    output_dim=params["output_dim"] 
    model_dim= params["model_dim"]
    mlp_dim=params["mlp_dim"]
    mlp_depth=params["mlp_depth"]
    drop_out=params["drop_out"]
    return_customized_layers=params["return_customized_layers"]
    loss=params["loss"]
    optimizer=params["optimizer"]
    metrics=params["metrics"]
    
    def _mlp_channel1(feature_dropout_layer, feature_map_layer, flatten_layer, x):
        x = feature_dropout_layer(x)
        x = feature_map_layer(x)
        x = flatten_layer(x)
        return x
    
    def _mlp_channel2(feature_map_layer, x):
        x = feature_map_layer(x)
        return x

    # inputs
    inputs1 = list()
    for fi in range(num_feature_channels1):
        inputs1.append(models.Input(shape=(num_features1, feature_dim1), dtype='float32', name='input1_' + str(fi)))
        
    inputs2 = list()
    for fi in range(num_feature_channels2):
        inputs2.append(models.Input(shape=(num_features2, ), dtype='float32', name='input2_' + str(fi)))
    
    # define feature map layers
    # MLP Layers
    feature_dropout_layer1 = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="input_dropout_layer"))
    feature_map_layer1 = layers.TimeDistributed(layers.Dense(model_dim, name="feature_map_layer1", activation="relu"))
    flatten_layer1 = layers.Flatten(name="feature_flatten_layer1")
    feature_map_layer2 = layers.Dense(model_dim, name="feature_map_layer2", activation="relu")
    
    x1 = [_mlp_channel1(feature_dropout_layer1, feature_map_layer1, flatten_layer1, input_) for input_ in inputs1]
    x2 = [_mlp_channel2(feature_map_layer2, input_) for input_ in inputs2]
    
    x = layers.Concatenate(axis=1, name="concate_layer")(x1+x2)
    
    # MLP Layers
    x = layers.BatchNormalization(name='batch_norm_layer')(x)
    x = layers.Dropout(rate=drop_out, name="dropout_layer")(x)
        
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model(inputs1 + inputs2, outputs)

    # compile model
    # adam = ko.Nadam()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if return_customized_layers:
        return model, {}

    return model

def build_multi_channel_cnn_model(params):    
    """
    Create A Multi-Layer Perceptron Model.
    
    inputs: 
        embeddings: [batch, num_embed_feature, embed_dims] * 3 ## pronoun, A, B
        positional_features: [batch, num_pos_feature] * 2 ## pronoun-A, pronoun-B
        
    outputs: 
        [batch, num_classes] # in our case there should be 3 output classes: A, B, None
        
    :param output_dim: the output dimension size
    :param num_filters: list of integers
        The number of filters.
    :param filter_sizes: list of integers
        The kernel size.
    :param pooling: str, either 'max' or 'average'
        Pooling method.
    :param padding: One of "valid", "causal" or "same" (case-insensitive).
        Padding method.
    :param model_dim: rrn dimension size
    :param mlp_dim: the dimension size of fully connected layer
    :param mlp_depth: the depth of fully connected layers
    :param drop_out: dropout rate of fully connected layers
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
    :return: keras model
    """

    num_feature_channels1=params["num_feature_channels1"] # number of channels in embedding features
    num_feature_channels2=params["num_feature_channels2"] # number of channels in positional features
    num_features1=params["num_features1"] # number of embeding features
    num_features2=params["num_features2"] # number of positional features
    feature_dim1=params["feature_dim1"] # embeding dimensions
    output_dim=params["output_dim"] 
    model_dim= params["model_dim"]
    mlp_dim=params["mlp_dim"]
    mlp_depth=params["mlp_depth"]
    drop_out=params["drop_out"]
    return_customized_layers=params["return_customized_layers"]
    loss=params["loss"]
    optimizer=params["optimizer"]
    metrics=params["metrics"]
    num_filters=params["num_filters"]
    filter_sizes=params["filter_sizes"]
    pooling=params["pooling"]
    padding=params["padding"]
        
    
    def _mlp_channel1(feature_dropout_layer, cnns, pools, concate_layer1, x):
        x = feature_dropout_layer(x)
        pooled_outputs = []
        for i in range(len(cnns)):
            conv = cnns[i](x)
            if pooling == 'max':
                conv = pools[i](conv)
            else:
                conv = pools[i](conv)
            pooled_outputs.append(conv)
        
        if len(cnns) == 1:
            x = conv
        else:
            x = concate_layer1(pooled_outputs)
        return x
    
    def _mlp_channel2(feature_map_layer, x):
        x = feature_map_layer(x)
        return x

    # inputs
    inputs1 = list()
    for fi in range(num_feature_channels1):
        inputs1.append(models.Input(shape=(num_features1, feature_dim1), dtype='float32', name='input1_' + str(fi)))
        
    inputs2 = list()
    for fi in range(num_feature_channels2):
        inputs2.append(models.Input(shape=(num_features2, ), dtype='float32', name='input2_' + str(fi)))
    
    # define feature map layers
    # CNN Layers
    cnns = []
    pools = []
    feature_dropout_layer1 = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="input_dropout_layer"))
    for i in range(len(filter_sizes)):
        cnns.append(layers.Conv1D(num_filters[i], kernel_size=filter_sizes[i], padding=padding, activation='relu', name="cc_layer1" + str(i)))
        if pooling == 'max':
            pools.append(layers.GlobalMaxPooling1D(name='global_pooling_layer1' + str(i)))
        else:
            pools.append(layers.GlobalAveragePooling1D(name='global_pooling_layer1' + str(i)))
    concate_layer1 = layers.Concatenate(name='concated_layer')
    
    feature_map_layer2 = layers.Dense(model_dim, name="feature_map_layer2", activation="relu")
    
    x1 = [_mlp_channel1(feature_dropout_layer1, cnns, pools, concate_layer1, input_) for input_ in inputs1]
    x2 = [_mlp_channel2(feature_map_layer2, input_) for input_ in inputs2]
    
    x = layers.Concatenate(axis=1, name="concate_layer")(x1+x2)
    
    # MLP Layers
    x = layers.BatchNormalization(name='batch_norm_layer')(x)
    x = layers.Dropout(rate=drop_out, name="dropout_layer")(x)
        
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model(inputs1 + inputs2, outputs)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if return_customized_layers:
        return model, {}

    return model

def _dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class RemappedCoAttentionWeight(merge._Merge):
    """
        Unnormalized Co-Attention operation for temporal data.
        Supports Masking.
        Follows the work of Ankur et al. [https://aclweb.org/anthology/D16-1244]
        "A Decomposable Attention Model for Natural Language Inference"
        # Input shape
            List of 2 3D tensor with shape: `(samples, steps1, features1)` and `(samples, steps2, features2)`.
        # Output shape
            3D tensor with shape: `(samples, steps1, step2)`.
        :param kwargs:
        """

    def __init__(self, model_size, activation='sigmoid',
                 W1_regularizer=None,  b1_regularizer=None,
                 W1_constraint=None, b1_constraint=None,
                 bias1=True, **kwargs):

        self.model_size = model_size
        self.init = initializers.get('glorot_uniform')

        self.W1_regularizer = regularizers.get(W1_regularizer)
        self.b1_regularizer = regularizers.get(b1_regularizer)

        self.W1_constraint = constraints.get(W1_constraint)
        self.b1_constraint = constraints.get(b1_constraint)

        self.bias1 = bias1
        self.activation = activations.get(activation)
        super(RemappedCoAttentionWeight, self).__init__(**kwargs)

    def build(self, input_shape):

        super(RemappedCoAttentionWeight, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError("input must be a size two list which contains two tensors")

        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        self.W1 = self.add_weight((self.model_size, shape1[-1]),
                                 initializer=self.init,
                                 name='{}_W1'.format(self.name),
                                 regularizer=self.W1_regularizer,
                                 constraint=self.W1_constraint)

        self.W2 = self.W1

        if self.bias1:
            self.b1 = self.add_weight((self.model_size,),
                                     initializer='zero',
                                     name='{}_b1'.format(self.name),
                                     regularizer=self.b1_regularizer,
                                     constraint=self.b1_constraint)

        if self.bias1:
            self.b2 = self.b1

    def compute_mask(self, input, input_mask=None):
        # pass the mask to the next layers
        return input_mask

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')

        x1, x2 = inputs[0], inputs[1]

        # u = Wx + b
        u1 = _dot_product(x1, self.W1)
        if self.bias1:
            u1 += self.b1

        u2 = _dot_product(x2, self.W2)
        if self.bias1:
            u2 += self.b2

        # u = Activation(Wx + b)
        u1 = self.activation(u1)
        u2 = self.activation(u2)

        # atten = exp(u1 u2^T)
        atten = K.batch_dot(u1, u2, axes=[2, 2])
        atten = K.exp(atten)

        return atten

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        if shape1[0] != shape2[0]:
            raise ValueError("batch size must be same")

        return shape1[0], shape1[1], shape2[1]

    def get_config(self):
        config = {
            'activation': self.activation,
            'model_size': self.model_size,
            'W1_regularizer': regularizers.serialize(self.W1_regularizer),
            'b1_regularizer': regularizers.serialize(self.b1_regularizer),
            'W1_constraint': constraints.serialize(self.W1_constraint),
            'b1_constraint': constraints.serialize(self.b1_constraint),
            'bias1': self.bias1,
        }
        base_config = super(RemappedCoAttentionWeight, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeatureNormalization(Layer):
    """
        Normalize feature along a specific axis.
        Supports Masking.

        # Input shape
            A ND tensor with shape: `(samples, feature1 ... featuresN).
        # Output shape
            ND tensor with shape: `(samples, feature1 ... featuresN)`.
        :param kwargs:
        """

    def __init__(self, axis=-1, **kwargs):

        self.axis = axis
        self.supports_masking = True
        super(FeatureNormalization, self).__init__(**kwargs)

    def build(self, input_shape):

        super(FeatureNormalization, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # don't pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a = K.cast(mask, K.floatx()) * inputs
        else:
            a = inputs

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=self.axis, keepdims=True) + K.epsilon(), K.floatx())
        
        return a

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis
        }
        base_config = super(FeatureNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_inter_coattention_cnn_model(params):
    """
    Create A Multi-Layer Perceptron Model with Coattention Mechanism.
    
    inputs: 
        embeddings: [batch, num_embed_feature, embed_dims] * 3 ## pronoun, A, B
        positional_features: [batch, num_pos_feature] * 2 ## pronoun-A, pronoun-B
        
    outputs: 
        [batch, num_classes] # in our case there should be 3 output classes: A, B, None
        
    :param output_dim: the output dimension size
    :param model_dim: rrn dimension size
    :param mlp_dim: the dimension size of fully connected layer
    :param mlp_depth: the depth of fully connected layers
    :param drop_out: dropout rate of fully connected layers
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
    :return: keras model
    """
    num_feature_channels1=params["num_feature_channels1"] # number of channels in embedding features
    num_feature_channels2=params["num_feature_channels2"] # number of channels in positional features
    num_features1=params["num_features1"] # number of embeding features
    num_features2=params["num_features2"] # number of positional features
    feature_dim1=params["feature_dim1"] # embeding dimensions
    output_dim=params["output_dim"] 
    model_dim= params["model_dim"]
    mlp_dim=params["mlp_dim"]
    mlp_depth=params["mlp_depth"]
    drop_out=params["drop_out"]
    return_customized_layers=params["return_customized_layers"]
    loss=params["loss"]
    optimizer=params["optimizer"]
    metrics=params["metrics"]
    num_filters=params["num_filters"]
    filter_sizes=params["filter_sizes"]
    pooling=params["pooling"]
    padding=params["padding"]
    atten_dim=params["atten_dim"]
    
    def _mlp_channel1(feature_dropout_layer, x):
        #x = feature_dropout_layer(x)
        return x
    
    def _mlp_channel2(feature_map_layer, x):
        x = feature_map_layer(x)
        return x

    # inputs
    inputs1 = list()
    for fi in range(num_feature_channels1):
        inputs1.append(models.Input(shape=(num_features1, feature_dim1), dtype='float32', name='input1_' + str(fi)))
        
    inputs2 = list()
    for fi in range(num_feature_channels2):
        inputs2.append(models.Input(shape=(num_features2, ), dtype='float32', name='input2_' + str(fi)))
    
    # define feature map layers
    # MLP Layers
    feature_dropout_layer1 = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="input_dropout_layer"))
    feature_map_layer2 = layers.Dense(feature_dim1, name="feature_map_layer2", activation="relu")
    
    x1 = [_mlp_channel1(feature_dropout_layer1, input_) for input_ in inputs1]
    x2 = [_mlp_channel2(feature_map_layer2, input_) for input_ in inputs2]
    
    # From mention-pair embeddings
    reshape_layer = layers.Reshape((1, feature_dim1), name="reshape_layer")
    x2 = [reshape_layer(x2_) for x2_ in x2]
    pair1 = layers.Concatenate(axis=1, name="concate_pair1_layer")([x1[0], x1[1], x2[0]])
    pair2 = layers.Concatenate(axis=1, name="concate_pair2_layer")([x1[0], x1[2], x2[1]])
    
    coatten_layer = RemappedCoAttentionWeight(atten_dim, name="coattention_weights_layer")
    featnorm_layer1 = FeatureNormalization(name="normalized_coattention_weights_layer1", axis=1)
    featnorm_layer2 = FeatureNormalization(name="normalized_coattention_weights_layer2", axis=2)
    focus_layer1 = layers.Dot((1, 1), name="focus_layer1")
    focus_layer2 = layers.Dot((2, 1), name="focus_layer2")
    pair_layer1 = layers.Concatenate(axis=-1, name="pair_layer1")
    pair_layer2 = layers.Concatenate(axis=-1, name="pair_layer2")
    
    # attention
    attens = coatten_layer([pair1, pair2])
    attens1 = featnorm_layer1(attens)
    attens2 = featnorm_layer2(attens)
    # compare
    focus1 = focus_layer1([attens1, pair1])
    focus2 = focus_layer2([attens2, pair2])
    pair1 = pair_layer1([pair1, focus2])
    pair2 = pair_layer2([pair2, focus1])
    
    x = layers.Concatenate(axis=1, name="concate_layer")([pair1, pair2])
    x = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="pair_dropout_layer"))(x)
    x = layers.TimeDistributed(layers.Dense(mlp_dim, name="pair_feature_map_layer", activation="relu"))(x)
    x = layers.Flatten(name="pair_feature_flatten_layer1")(x)
    
#     pooled_outputs = []
#     for i in range(len(filter_sizes)):
#         conv = layers.Conv1D(num_filters[i], kernel_size=filter_sizes[i], padding=padding, activation='relu')(x)
#         if pooling == 'max':
#             conv = layers.GlobalMaxPooling1D(name='global_pooling_layer' + str(i))(conv)
#         else:
#             conv = layers.GlobalAveragePooling1D(name='global_pooling_layer' + str(i))(conv)
#         pooled_outputs.append(conv)
#     if len(pooled_outputs) > 1:
#         x = layers.Concatenate(name='concated_layer')(pooled_outputs)
#     else:
#         x = conv
    
    # MLP Layers
    x = layers.BatchNormalization(name='batch_norm_layer')(x)
    x = layers.Dropout(rate=drop_out, name="dropout_layer")(x)
        
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model(inputs1 + inputs2, outputs)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    if return_customized_layers:
        return model, {'RemappedCoAttentionWeight': RemappedCoAttentionWeight,
                       "FeatureNormalization": FeatureNormalization}

    return model


def build_intra_coattention_cnn_model(params):
    """
    Create A Multi-Layer Perceptron Model with Coattention Mechanism.
    
    inputs: 
        embeddings: [batch, num_embed_feature, embed_dims] * 3 ## pronoun, A, B
        positional_features: [batch, num_pos_feature] * 2 ## pronoun-A, pronoun-B
        
    outputs: 
        [batch, num_classes] # in our case there should be 3 output classes: A, B, None
        
    :param output_dim: the output dimension size
    :param model_dim: rrn dimension size
    :param mlp_dim: the dimension size of fully connected layer
    :param mlp_depth: the depth of fully connected layers
    :param drop_out: dropout rate of fully connected layers
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
    :return: keras model
    """
    num_feature_channels1=params["num_feature_channels1"] # number of channels in embedding features
    num_feature_channels2=params["num_feature_channels2"] # number of channels in positional features
    num_features1=params["num_features1"] # number of embeding features
    num_features2=params["num_features2"] # number of positional features
    feature_dim1=params["feature_dim1"] # embeding dimensions
    output_dim=params["output_dim"] 
    model_dim= params["model_dim"]
    mlp_dim=params["mlp_dim"]
    mlp_depth=params["mlp_depth"]
    drop_out=params["drop_out"]
    return_customized_layers=params["return_customized_layers"]
    loss=params["loss"]
    optimizer=params["optimizer"]
    metrics=params["metrics"]
    num_filters=params["num_filters"]
    filter_sizes=params["filter_sizes"]
    pooling=params["pooling"]
    padding=params["padding"]
    atten_dim=params["atten_dim"]
    
    
    def _mlp_channel1(feature_dropout_layer, x):
        #x = feature_dropout_layer(x)
        return x
    
    def _mlp_channel2(feature_map_layer, x):
        x = feature_map_layer(x)
        return x
    
    def coatten_compare(
        feature_concat_layer, coatten_layer, 
        featnorm_layer1, featnorm_layer2, 
        focus_layer1, focus_layer2, 
        pair_layer1, pair_layer2, 
        mention, entity, mention_entity_feature):
        
        x1 = feature_concat_layer([entity, mention_entity_feature])
        x2 = feature_concat_layer([mention, mention_entity_feature])
        
        # attention
        attens = coatten_layer([x1, x2])
        attens1 = featnorm_layer1(attens)
        attens2 = featnorm_layer2(attens)
        # compare
        focus1 = focus_layer1([attens1, x1])
        focus2 = focus_layer2([attens2, x2])
        x1 = pair_layer1([x1, focus2])
        x2 = pair_layer2([x2, focus1])
        
        return x1, x2

    # inputs
    inputs1 = list()
    for fi in range(num_feature_channels1):
        inputs1.append(models.Input(shape=(num_features1, feature_dim1), dtype='float32', name='input1_' + str(fi)))
        
    inputs2 = list()
    for fi in range(num_feature_channels2):
        inputs2.append(models.Input(shape=(num_features2, ), dtype='float32', name='input2_' + str(fi)))
    
    # define feature map layers
    # MLP Layers
    feature_dropout_layer1 = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="input_dropout_layer"))
    feature_map_layer2 = layers.Dense(feature_dim1, name="feature_map_layer2", activation="relu")
    
    x1 = [_mlp_channel1(feature_dropout_layer1, input_) for input_ in inputs1]
    x2 = [_mlp_channel2(feature_map_layer2, input_) for input_ in inputs2]
    
    # From mention-pair embeddings
    reshape_layer = layers.Reshape((1, feature_dim1), name="reshape_layer")
    x2 = [reshape_layer(x2_) for x2_ in x2]
    
    feature_concat_layer = layers.Concatenate(axis=1, name="concate_pair_layer")
    coatten_layer = RemappedCoAttentionWeight(atten_dim, name="coattention_weights_layer")
    featnorm_layer1 = FeatureNormalization(name="normalized_coattention_weights_layer1", axis=1)
    featnorm_layer2 = FeatureNormalization(name="normalized_coattention_weights_layer2", axis=2)
    focus_layer1 = layers.Dot((1, 1), name="focus_layer1")
    focus_layer2 = layers.Dot((2, 1), name="focus_layer2")
    pair_layer1 = layers.Concatenate(axis=-1, name="pair_layer1")
    pair_layer2 = layers.Concatenate(axis=-1, name="pair_layer2")
    
    pairs = list()
    
    pairs += list(coatten_compare(
        feature_concat_layer, coatten_layer,
        featnorm_layer1, featnorm_layer2, 
        focus_layer1, focus_layer2, 
        pair_layer1, pair_layer2, 
        x1[0], x1[1], x2[0]))
    
    pairs += list(coatten_compare(
        feature_concat_layer, coatten_layer,
        featnorm_layer1, featnorm_layer2, 
        focus_layer1, focus_layer2, 
        pair_layer1, pair_layer2, 
        x1[0], x1[2], x2[1]))
    
    x = layers.Concatenate(axis=1, name="concate_layer")(pairs)
    x = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="pair_dropout_layer"))(x)
    x = layers.TimeDistributed(layers.Dense(mlp_dim, name="pair_feature_map_layer", activation="relu"))(x)
    x = layers.Flatten(name="pair_feature_flatten_layer1")(x)
    
#     pooled_outputs = []
#     for i in range(len(filter_sizes)):
#         conv = layers.Conv1D(num_filters[i], kernel_size=filter_sizes[i], padding=padding, activation='relu')(x)
#         if pooling == 'max':
#             conv = layers.GlobalMaxPooling1D(name='global_pooling_layer' + str(i))(conv)
#         else:
#             conv = layers.GlobalAveragePooling1D(name='global_pooling_layer' + str(i))(conv)
#         pooled_outputs.append(conv)
#     if len(pooled_outputs) > 1:
#         x = layers.Concatenate(name='concated_layer')(pooled_outputs)
#     else:
#         x = conv
    
    # MLP Layers
    x = layers.BatchNormalization(name='batch_norm_layer')(x)
    x = layers.Dropout(rate=drop_out, name="dropout_layer")(x)
        
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model(inputs1 + inputs2, outputs)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if return_customized_layers:
        return model, {'RemappedCoAttentionWeight': RemappedCoAttentionWeight,
                       "FeatureNormalization": FeatureNormalization}

    return model


def train_test(model,params):
    
    model_name=params["model_name"]
    X_train=params["X_train"]
    y_train=params["y_train"]
    X_dev=params["X_dev"]
    y_dev=params["y_dev"]
    patience=params["patience"]
    epochs=params["epochs"]
    batch_size=params["batch_size"]

    weightFolder="./weights/"
    if not os.path.exists(weightFolder):
        os.mkdir(weightFolder)
        print(weightFolder+" created!")

    file_path = os.path.join(weightFolder,model_name+".hdf5")
    check_point = kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")
    early_stop = kc.EarlyStopping(monitor = "val_loss", mode = "min", patience=patience)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_dev, y_dev), callbacks = [check_point, early_stop],verbose=0)

    minLoss=np.min(np.asarray(history.history['val_loss']))
    return minLoss,file_path
        