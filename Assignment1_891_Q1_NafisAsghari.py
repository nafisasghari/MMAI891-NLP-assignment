# -*- coding: utf-8 -*-
"""
Nafis Asghari
Student Number: 20196621
MMAI
Cohort: 2020
Course: MMAI 891 - Natural Language Processing
June 15, 2020


Submission to Question 1 - Main file
"""

################################
### Import packages and Data ###
################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import re
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import os
import itertools
import argparse

in_dir = 'data'
out_dir = 'out/lexicon'

df_train = pd.read_csv(os.path.join(in_dir, "sentiment_train.csv"))

print(df_train.info())
print(df_train.head())

df_test = pd.read_csv(os.path.join(in_dir, "sentiment_test.csv"))

print(df_test.info())
print(df_test.head())


#############################
### Some utils functions ###
#############################

# convert sentiment scores to 0/1
def pred_polarity(x, threshold=0):
    if x > threshold:
        return 1
    else:
        return 0


# normalization on sentence level
def preprocess_sent(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # [.?!] --> [.?!] xxx
    s = re.sub(r'([.?!"])(\w)', r'\1 \2', s)
    s = re.sub(r'(\w)(")', r'\1 \2', s)
    s = re.sub('waw', 'what a waste', s, flags=re.IGNORECASE)

    return s.strip()


########################################
### LEXICON-BASED APPROACH functions ###
########################################

# add and modify some words and their scores on vader dictionary
new_words = {
    'prompt': 3, 'happy': 3, 'inexpensive': 2, 'tender': 3, 'moist': 3, 'tasty': 3, 'fresh': 3, 'star': 2, 'back': 2,
    'quick': 3, 'extraordinary': 3,
    'phenomenal': 3, 'solid': 2, 'immediate': 2, 'check it out': 4, 'must have': 4, 'see': 4, 'watch': 4, 'touch': 2,
    'skilled': 3, '10': 4,
    'slow': -3, 'expensive': -2, 'dry': -3, 'overpriced': -3, 'away': -4, 'sloppy': -3, 'lame': -3, 'wait': -2, '1': -3,
    'worst': -5, 'garbage': -4
}


def lexicon_based_sentiment_analysis(df, text_col, polarity_col=None, preprocess=True,
                                     add_new_words=True, threshold=0, save_wrong_pred=False,
                                     save_df=False, for_plot=False):

    # Instantiate new SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    if add_new_words:
        sid.lexicon.update(new_words)

    # Generate sentiment scores
    if preprocess:
        sentiment = df[text_col].apply(preprocess_sent).apply(sid.polarity_scores)

    else:
        sentiment = df[text_col].apply(sid.polarity_scores)

    # Generate sentiment scores
    df['sentiment_scores'] = pd.Series([s['compound'] for s in sentiment], index=sentiment.index)

    df['pred_polarity'] = df['sentiment_scores'].apply(lambda x: pred_polarity(x, threshold))

    if save_df:
        df.to_csv(os.path.join(out_dir, 'df_preds_lexicon_based.csv'))

    if polarity_col is not None:

        acc = accuracy_score(df[polarity_col], df['pred_polarity'])
        f1 = f1_score(df[polarity_col], df['pred_polarity'])

        #     pd.set_option('display.max_colwidth', -1)
        if save_wrong_pred:
            print(f'preprocess_sent: {preprocess} , add_new_words: {add_new_words} , threshold: {threshold}')
            cm = confusion_matrix(df[polarity_col], df['pred_polarity'])
            print(cm)
            print('accuracy: ', acc)
            print('f1, ', f1)
            print('')

            wrong_preds = df[df[polarity_col] != df.pred_polarity]
            wrong_preds.to_csv(
                os.path.join(out_dir, f'wrong_preds_lexicon_pre_{preprocess}_add_{add_new_words}_{threshold}.csv'))
            print('Wrong predictions:')
            print(wrong_preds.head(5))

        if for_plot:
            return acc, f1

    return


def plot_acc_f1_scores(df, preprocess=[True, False], add_words=[True, False],
                       threshold_range=np.arange(-0.5, 0.55, 0.05)):
    accuracies = []
    f1s = []
    vars_list = list(itertools.product(threshold_range, preprocess, add_words))

    for thrsh, pre, new_word in tqdm(vars_list):
        #     for thrsh in tqdm(threshold_range):

        acc, f1 = lexicon_based_sentiment_analysis(df, text_col='Sentence', polarity_col='Polarity',
                                                   preprocess=pre, add_new_words=new_word,
                                                   threshold=thrsh, for_plot=True)

        accuracies.append(acc)
        f1s.append(f1)

    vars_name = list(itertools.product(['%.2f' % t for t in threshold_range], preprocess, add_words))
    pd.DataFrame({'Parameters': vars_name, 'acc': accuracies, 'f1': f1s}).to_csv(
        os.path.join(out_dir, 'metrics_lexicon_based.csv'))
    plt.figure(figsize=(10, 5))
    plt.title('Accuracy and F1 scores')
    plt.plot(range(len(vars_list)), accuracies, label='acc')
    plt.plot(range(len(vars_list)), f1s, label='f1')
    plt.xticks(range(len(vars_list)), vars_name, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'acc_f1_plot.png'))
    plt.show()

    return


#---------------

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='Train, Test, Trainplot, Testplot', default='Trainplot')
args = parser.parse_args()


if __name__ == "__main__":

    if args.mode == 'Trainplot':
        plot_acc_f1_scores(df_train, preprocess=[True, False], add_words=[True, False],
                           threshold_range=np.arange(-0.15, 0.3, 0.05))

    if args.mode == 'Testplot':
        plot_acc_f1_scores(df_test, preprocess=[True, False], add_words=[True, False],
                           threshold_range=np.arange(-0.15, 0.30, 0.05))

    if args.mode == 'Train':
        lexicon_based_sentiment_analysis(df_train, text_col='Sentence', polarity_col='Polarity',
                                         preprocess=True, add_new_words=True, threshold=0.1,
                                         save_wrong_pred=True, save_df=True)

    if args.mode == 'Test':
        lexicon_based_sentiment_analysis(df_test, text_col='Sentence', polarity_col='Polarity',
                                         preprocess=True, add_new_words=True, threshold=0.1,
                                         save_wrong_pred=True, save_df=True)
