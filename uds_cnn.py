#!/usr/bin/python
# encoding=utf8
# refers to:
#       https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
#       http://machinelearningmastery.com/save-load-keras-deep-learning-models/

from __future__ import print_function
import os
import numpy as np
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.models import model_from_json
from keras.regularizers import l2, activity_l2
import six.moves.cPickle

TRAIN_MODE = False
NB_EPOCH = 2
BASE_DIR = 'mixed_ds_train'
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
DATAFILES = ['reviews_rt_all.csv','imdb_small.csv']   # dataset for verification should be here
TOKENIZER = "cnn_tokenizer"

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {'negative': 0, 'positive': 1}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

for df in DATAFILES:
    with open(df) as text_samples:
        for line in text_samples:
            tvalues = line.split('|')
            lbl = tvalues[0]
            txt = tvalues[1].lower()
            txt = re.sub("[^a-z-_.\s]", u'', txt)
            if lbl.isdigit() and txt:
                labels.append(lbl)
                texts.append(txt)

print('Found %s texts.' % len(texts))

if (TRAIN_MODE):
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    six.moves.cPickle.dump(tokenizer, open(TOKENIZER, "wb"))
else:
    tokenizer = six.moves.cPickle.load(open(os.path.join(BASE_DIR, TOKENIZER), 'rb'))

sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
# print(data[:1])
print('Shape of label tensor:', labels.shape)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(data, labels,
                                                  test_size=VALIDATION_SPLIT,
                                                  random_state=42, stratify=labels)

if (TRAIN_MODE):
    print('Preparing embedding matrix.')

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)
    print('Training model.')

# train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
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

    model = Model(sequence_input, preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

# happy learning!
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch=NB_EPOCH, batch_size=128)

    print("Save model to disk")

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("model.h5")
    print("complete")

else:
# load json and create model
    json_file = open(os.path.join(BASE_DIR, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights(os.path.join(BASE_DIR, 'model.h5'))
    print("Loaded model from disk")

# evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # score = loaded_model.evaluate(x_val, y_val)
    # score = loaded_model.evaluate(x_train, y_train)
    score = loaded_model.evaluate(data, labels)
    print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
