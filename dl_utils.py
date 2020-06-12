# -*- coding: utf-8 -*-
"""
Nafis Asghari
Student Number: 20196621
MMAI
Cohort: 2020
Course: MMAI 891 - Natural Language Processing
June 15, 2020


Submission to Question 3 - Deep learning architectures and Evaluation functions
"""

################################
### Import packages and Data ###
################################
from nlp_preprocessing import text_preprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import spacy

from keras import backend as K
from keras.layers import Dense, Input, LSTM, Activation, Conv1D, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers.merge import add
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score, log_loss, f1_score, precision_recall_curve


nlp = spacy.load('en_core_web_md')


##################################
# ---------Deep Learning---------#
##################################
def keras_preprocess(sentences, maxlen):
    # Clean text
    clean_sents = sentences.apply(text_preprocess)

    # tokenizing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean_sents)
    sequences = tokenizer.texts_to_sequences(clean_sents)

    dl_data = sequence.pad_sequences(sequences, maxlen=maxlen)

    word_index = tokenizer.word_index

    EMBEDDING_DIM = 300
    embeddings_index = {}

    for word, idx in word_index.items():
        try:
            embedding = nlp(word).vector
            embeddings_index[word] = embedding
        except:
            pass

    print('Found %s unique tokens.' % len(word_index))
    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=maxlen,
                                trainable=False)

    return np.array(dl_data), embedding_layer


def cnn(maxlen, embedding_layer):
    # 1D CNN
    inp = Input(shape=(maxlen,), dtype='int32')
    embedding = embedding_layer(inp)
    stacks = []
    for kernel_size in [2, 3, 4]:
        conv = Conv1D(64, kernel_size, padding='same', activation='relu', strides=1)(embedding)
        pool = MaxPooling1D(pool_size=3)(conv)
        drop = Dropout(0.5)(pool)
        stacks.append(drop)

    merged = Concatenate()(stacks)
    flatten = Flatten()(merged)
    drop = Dropout(0.5)(flatten)
    outp = Dense(2, activation='sigmoid')(drop)

    TextCNN = Model(inputs=inp, outputs=outp)
    TextCNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    TextCNN.summary()

    return TextCNN


def lstm(maxlen, embedding_layer):
    Lstm = Sequential()

    inp = Input(shape=(maxlen,), dtype='int32')
    Lstm.add(embedding_layer)
    Lstm.add(LSTM(300, dropout=0.3))
    Lstm.add(Dense(2, activation='sigmoid'))
    Lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    Lstm.summary()
    return Lstm


def dl_model(model, maxlen, embedding_layer, x, y, out_dir=None):
    if model == 'Lstm':
        clf = lstm(maxlen, embedding_layer)
    elif model == 'CNN':
        clf = cnn(maxlen, embedding_layer)
    else:
        raise ValueError('model is invalid')

    early_stopping_monitor = EarlyStopping(patience=3)

    history = clf.fit(x,
                      y,
                      batch_size=32,
                      epochs=20,
                      validation_split=0.3,
                      #                       validation_data=(x_test, y_test),
                      callbacks=[early_stopping_monitor])

    train_val_plots(model_history=history, clf_name=model, out_dir=out_dir)

    return clf


##############################
# Model Evaluation functions #
##############################
def train_val_plots(model_history, clf_name=None, out_dir=None):
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f'Training_validation_acc_{clf_name}.png'))

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f'Training_validation_loss_{clf_name}.png'))


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, clf_name=None, out_dir=None):
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.ylim([0, 1])
    plt.savefig(os.path.join(out_dir, f'precision_recall_plot_{clf_name}.png'))


def metrics(df_test, clf, x_test, y_test, thrsh=0.5, clf_name=None, out_dir=None):
    predicted = clf.predict(x_test)

    y_proba = [y[1] for y in predicted]
    y_pred = [np.where(x >= thrsh, 1, 0) for x in y_proba]
    y_test = y_test.argmax(axis=1)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("-----" * 5)
    print(classification_report(y_test, y_pred))

    print("-----" * 5)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {0:.3f}".format(acc))

    kappa = cohen_kappa_score(y_test, y_pred)
    print("Kappa: {0:.3f}".format(kappa))

    logloss = log_loss(y_test, y_proba)
    print("Log loss: {0:.3f}".format(logloss))

    f1 = f1_score(y_test, y_pred)
    print("f1: {0:.3f}".format(f1))

    metrics_df = pd.DataFrame({"Accuracy": [acc], "f1": [f1], "Kappa": [kappa], 'log_loss': [logloss]}).round(3)
    metrics_df.to_csv(os.path.join(out_dir, f'metrics_dl_based_{clf_name}.csv'), index=False)

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds, clf_name, out_dir)

    df_test['y_pred'] = y_pred
    df_test['y_proba'] = y_proba
    df_test.to_csv(os.path.join(out_dir, f'df_preds_dl_based_{clf_name}.csv'))

    wrong_preds = df_test[df_test.Polarity != df_test.y_pred]

    wrong_preds.to_csv(os.path.join(out_dir, f'wrong_preds_dl_{thrsh}_test_{clf_name}.csv'))

    return


if __name__ == '__main__':
    pass
