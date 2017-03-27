#!/usr/bin/python
# encoding=utf8
# refers to:
#       https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
#       http://machinelearningmastery.com/save-load-keras-deep-learning-models/

from __future__ import print_function
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import re
import time
import datetime

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, GlobalAveragePooling1D
from keras.models import Model, Sequential, model_from_json
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras import backend as K

from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import words
from nltk import wordpunct_tokenize as wt
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

import six.moves.cPickle

TRAIN_MODE = True
DATA_DIR = 'data'
NB_EPOCH = 50
BASE_DIR = 'test_models'  # saved models for test
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.1
DATAFILES = 'train_movies.csv'  # dataset for verification should be here
DATA_PROPORTIONS = [1, 1.5, 2.5, 5, 6]
TOKENIZER = "cnn_tokenizer"
RANDOM_SEED = 42
METRICS = ['accuracy', 'fmeasure', 'precision', 'recall']
# second, prepare text samples and their labels
print('Processing text dataset')

# texts = []  # list of text samples
labels_index = {'negative': 0, 'positive': 1}  # dictionary mapping label name to numeric id
# labels = []  # list of label ids

cached_stopwords = ['href','quot','amp','br',
                    'an','by','did','does','was',
                    'were','the','to','at','on',
                    'in','with','it','he','she',
                    'this','that','is']


def load_test_dataset():
    positive_examples = pd.read_csv('./test/rt-polarity_pos.txt', sep='|', encoding='latin-1')
    negative_examples = pd.read_csv('./test/rt-polarity_neg.txt', sep='|', encoding='latin-1')
    positive_examples['overall'] = np.ones(shape=(len(positive_examples), 1), dtype=int)
    negative_examples['overall'] = np.zeros(shape=(len(negative_examples), 1), dtype=int)
    dfr = pd.concat([positive_examples, negative_examples], ignore_index=True)
    return dfr


stem = PorterStemmer() #  if len(w) > 2 NLTK bug (Rewrite porter.py #1261)


def stemmer(line, stem=stem):
    return " ".join([stem.stem(w) for w in line.split()]) 


def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in cached_stopwords])


def remove_names(text):
    tagged_sent = pos_tag(text.split())
    res = ' '.join([word for word, pos in tagged_sent if pos != 'NNP'])
    return res.lower()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"can\'t", "can not", str(string))
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\b[A-Za-z]{1}\b", ' ', string)
    string = re.sub(r"[^A-Za-z-_]", " ", string)
    string = re.sub(r'\.{1,10}', ' ', string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_data(pos_ratio):
    df_train = pd.read_csv(os.path.join(DATA_DIR, DATAFILES))
    print("dataset loaded successfully")
    df_train_neg = df_train[(df_train.overall == 0) & (~df_train.reviewText.isnull())].copy()
    # df_train_neg = df_train_neg[:50000]
    n = len(df_train_neg)
    print('Found %s complaints.' % n)
    df_train_pos, _ = train_test_split(df_train[(df_train.overall == 1) & (~df_train.reviewText.isnull())],
                                       train_size=int(n*pos_ratio), random_state=RANDOM_SEED)
    df_train = pd.concat([df_train_pos, df_train_neg])
    print('Found %s texts.' % len(df_train))
    # df_train = df_train.sample(frac=1).reset_index(drop=True)
    return df_train


def process_train_df(gdhs, preprocessor=stemmer):
    print('--data processing--')
    its1 = datetime.datetime.now()
    gdhs['new_text'] = gdhs['reviewText'].apply(clean_str)
    its2 = datetime.datetime.now()
    print('cleaning:', its2 - its1)
    # gdhs['new_text'] = gdhs['new_text'].apply(remove_names)
    its1 = datetime.datetime.now()
    print('remove_names:', its1 - its2)
    gdhs['new_text'] = gdhs['new_text'].apply(remove_stopwords)
    its2 = datetime.datetime.now()
    print('remove_stopwords:', its2 - its1)
    # gdhs['new_text'] = gdhs['new_text'].apply(preprocessor)
    its1 = datetime.datetime.now()
    print('stemmer:', its1 - its2)
    gdhs = gdhs.reset_index()
    gdhs = gdhs[gdhs['new_text'].notnull()]
    gdhs = gdhs.ix[:, ['overall', 'new_text', 'asin']]
    its2 = datetime.datetime.now()
    print('reshaping:', its2 - its1)
    return gdhs


def split_data(df_train):
    train, test = train_test_split(df_train.asin.unique(), test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    train_reviews, labels_train = df_train.loc[df_train.asin.isin(train), "new_text"].values, df_train.loc[df_train.asin.isin(train), "overall"].values
    test_reviews, labels_test = df_train.loc[df_train.asin.isin(test), "new_text"].values, df_train.loc[df_train.asin.isin(test), "overall"].values
    labels_train_reviews = to_categorical(np.asarray(labels_train))
    labels_test_reviews = to_categorical(np.asarray(labels_test))
    # do I really need this piece of code?
    # indexes = np.arange(len(train_reviews))
    # np.random.seed(RANDOM_SEED)
    # np.random.shuffle(indexes)
    # train_reviews, labels_train_reviews = train_reviews[indexes], labels_train_reviews[indexes]
    return train_reviews, test_reviews, labels_train_reviews, labels_test_reviews


def create_model(layer):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(7)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = Dense(128, input_dim=128, W_regularizer=l2(0.01),
              activity_regularizer=activity_l2(0.01))(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(len(labels_index), activation='softmax')(x)
    return Model(sequence_input, preds)


def create_seq_model(layer):
    model = Sequential()
    model.add(layer)
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(len(labels_index), activation='sigmoid'))
    return model


def train_model(coefficient_of_asymmetry):
    suffix = str(datetime.datetime.now().isoformat())
    tensorboard_cb = TensorBoard(log_dir='./logs/logs_{}'.format(suffix), histogram_freq=0,
                                 write_graph=False, write_images=False)
    stopper_cb = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=1, mode='auto')
    checkpoint_cb = ModelCheckpoint("./checkpoint/weights_amazon_cnn.h5",
                                    monitor='val_loss', save_best_only=True, verbose=0)
    slower_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1,
                                  mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    csv_logger = CSVLogger("./logs/training_{}.log".format(suffix))
    ts1 = datetime.datetime.now()
    df = load_data(coefficient_of_asymmetry)
    ts2 = datetime.datetime.now()
    print('load_data:', ts2 - ts1)
    df = df.sample(frac=1).reset_index(drop=True)
    df = process_train_df(df)
    ts1 = datetime.datetime.now()
    print('process_train_df:', ts1 - ts2)
    # df.to_csv("./data/processed_train_movies_5050.csv", sep='|', index=False)

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df["new_text"])
    six.moves.cPickle.dump(tokenizer, 
        open("./tokenizers/{}_{}_{}".format(TOKENIZER, coefficient_of_asymmetry,
                                            str(suffix)), "wb"))
    word_index = tokenizer.word_index
    ts2 = datetime.datetime.now()
    print('tokenization:', ts2 - ts1)
    print('Found %s unique tokens.' % len(word_index))

    x_train_text, x_val_text, y_train, y_val = split_data(df)

    sequences_train = tokenizer.texts_to_sequences(x_train_text)
    x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    sequences_val = tokenizer.texts_to_sequences(x_val_text)
    x_val = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)

    ts1 = datetime.datetime.now()
    print('sequences:', ts1 - ts2)
    print('Preparing embedding matrix.')

    embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)
    ts2 = datetime.datetime.now()
    print('embeddings:', ts2 - ts1)
    print('Training model.')

# train a 1D convnet with global maxpooling
#     model = create_model(embedding_layer)
    model = create_seq_model(embedding_layer)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val), verbose=1,
              nb_epoch=NB_EPOCH, batch_size=128, shuffle=True,
              callbacks=[stopper_cb, checkpoint_cb, slower_cb, csv_logger, tensorboard_cb])
    ts1 = datetime.datetime.now()
    print('model creation and fitting:', ts1 - ts2)
    print("Save model to disk")

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    K.clear_session()  # TF bug: MemLeak;Exception ignored in BaseSession.__del__ #3388
    print("complete")

tokenizers = ['cnn_tokenizer_1','cnn_tokenizer_2','cnn_tokenizer_3','cnn_tokenizer_4']
weights = ['weights00-08708-03134.h5', 'weights01-08831-02930.h5','weights01-08927-02670.h5', 'weights01-09266-01895.h5']
models = ['model1.json', 'model2.json', 'model3.json', 'model4.json']

if TRAIN_MODE:
    for i in DATA_PROPORTIONS:
        train_model(i)
else:
    for i in DATA_PROPORTIONS:
        df = load_data(i)
        # df = load_test_dataset()
        df = process_train_df(df)
        print("dataset %f\n" % i)
        for j in range(0,4):
            print("iteration %d" % j)
            tokenizer = six.moves.cPickle.load(open(os.path.join(BASE_DIR, tokenizers[j]), 'rb'))
            
            sequences = tokenizer.texts_to_sequences(df['new_text'])

            data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            labels = to_categorical(np.asarray(df['overall']))
            # load json and create model
            json_file = open(os.path.join(BASE_DIR, models[j]), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(os.path.join(BASE_DIR, weights[j]))
            print("Loaded model from disk")

            # evaluate loaded model on test data
            loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
            # loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
            score = loaded_model.evaluate(data, labels)

            with open("eval_log.txt", 'a') as eval_data:
                eval_data.write('%s,%s\n' % (loaded_model.metrics_names[1], loaded_model.metrics_names[2]))
                eval_data.write("%.4f,%.4f\n" % (score[1], score[2]))

            print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        K.clear_session()
