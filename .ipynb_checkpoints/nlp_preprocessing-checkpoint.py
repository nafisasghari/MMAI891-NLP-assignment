# -*- coding: utf-8 -*-
"""
Nafis Asghari
Student Number: 20196621
MMAI
Cohort: 2020
Course: MMAI 891 - Natural Language Processing
June 15, 2020


Submission to Question 2 - NLP Preprocessing steps
"""

################################
### Import packages and Data ###
################################
import pandas as pd
import numpy as np
import re

from tqdm import tqdm
import os

import gensim
from gensim import models
import re
from language_detector import detect_language
import pkg_resources
from symspellpy import SymSpell, Verbosity
import unidecode

import string
import spacy
from spacy.lang.en import English

import textstat
from sklearn.feature_extraction.text import TfidfVectorizer


######################################
# ---------Text preprocessing---------#
######################################

# Spelling correction
sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)



#### Sentence level preprocess ####

# lowercase + base filter
# some basic normalization
def f_base(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 4: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)

    # normalization 5: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 6: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)
    # normalization 6: Remove non-unicode
    s = unidecode.unidecode(s)
    # normalization 7: Remove numbers
    s = re.sub(r'\d+', '', s)

    return s.strip()


# language detection
def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """

    # some reviews are french
    return detect_language(s) in {'English'}


#### word level preprocess ####

punctuations = string.punctuation
# filtering out punctuations
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word not in punctuations]


# typo correction
def f_typo(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
        else:
            pass
            # do word segmentation, deprecated for inefficiency
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # w_list_fixed.extend(w_seg.corrected_string.split())
    return w_list_fixed


def f_lemma(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    # SpaCy
    return [word.lemma_.strip() if word.lemma_ != "-PRON-" else word.lower_ for word in w_list]



# filtering out stop words 
stop_words_to_keep = ["no", "n't", "not"]
stop_words = [stop for stop in spacy.lang.en.stop_words.STOP_WORDS if stop not in stop_words_to_keep]

extend_stop_words = []

stop_words.extend(extend_stop_words)


def f_stopw(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in stop_words]



def preprocess_sent(rw):
    """
    Get sentence level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: sentence level pre-processed review
    """
    s = f_base(rw)
#     if not f_lan(s):
#         return None
    return s


parser = English()

def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed review
    """
    if not s:
        return None
    w_list = parser(s)
    w_list = f_lemma(w_list)
    w_list = f_punct(w_list)
    w_list = f_typo(w_list)
    w_list = f_stopw(w_list)

    return ' '.join(w_list)


def sent_to_words(sentences):
    for sent in sentences:
        sent = preprocess_sent(sent)
        sent = preprocess_word(sent)

        yield (sent)
        

def text_preprocess(sent):
        sent = preprocess_sent(sent)
        sent = preprocess_word(sent)

        return sent      
        

        
# Feature Engineering and Feature Extraction - before cleaning the text 
def feature_eng(df, text_col):
    org_cols = df.columns
    #length of original text
    df['Sent_length'] = df[text_col].apply(len)
    
    #more than 2 Exclamation mark (!!) - binary feature
    reg = re.compile("(!)\\1{1,}")
    has_more_than2_exm = lambda x : np.where(len(reg.findall(x)) == 0 , 0 , 1 )
    
    df['more_that_2_exm'] = df[text_col].apply(has_more_than2_exm)
    df['words_count'] = df[text_col].apply(lambda x: textstat.lexicon_count(x))
    num_cols = [col for col in df.columns if col not in org_cols]
    return num_cols

######################################
### Method 1:  Tf-idf, n-grams-BOW ###
######################################
def bow_tfidf(df , text_col , no_features = 1000 , train = True):
    
    if train:
        global vectorizer
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=0.005,
                                       max_features=no_features, ngram_range=[1,3])
        dtm = vectorizer.fit_transform(df[text_col])
        
    else:
        dtm = vectorizer.transform(df[text_col])
    
    bow_tfidf_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names(), index=df.index)
    
    return bow_tfidf_df
 

def merge_all_features(df, doc_vec_df, num_feats= []):
    df = df.drop(columns=[col for col in df.columns if col not in num_feats])
    merge_df = pd.concat([df, doc_vec_df], axis=1)
    return merge_df

    
def bow_preprocessing(df, text_col, target = None, feat_eng = True, train = True, no_features = 1000):   
    
    if feat_eng:
        num_feats = feature_eng(df, text_col)
    else:
        num_feats = []
    
    df['clean_text'] = df[text_col].apply(text_preprocess)
    
    bow_df = bow_tfidf(df , text_col = 'clean_text' , no_features = no_features, train = train)
    
    x = merge_all_features(df, bow_df, num_feats)
    
    if target is not None:       
        y = df[target]   
    else: 
        y = None
        
    return x, y


######################################
###  Method 2: Pre-Trained GloVe   ###
######################################
# Covert whole sentence to a vector
# import en_core_web_md
def text_to_vector(sentences, nlp):
    """
    :param series: pandas series of sentences
    :param nlp: SpaCy nlp object

    :return: Dataframe of vectors (length of input series , 300)
    """
    text_clean = sentences.apply(text_preprocess)
    vec = text_clean.apply(lambda x: nlp(x).vector)

    vectors = pd.DataFrame(vec.values.tolist())
    
    return vectors


def glove_preprocessing(df, col_to_vec, nlp , target = None, feat_eng=True):
    """
    :param nlp: spaCy nlp object
    :param df: DataFrame
    :param col_to_vec: text column
    :param target: string, target columns names from df
    :param cat_feats: list of strings, categorical features name from df

    :return: DataFrame , series( if target is not null)
    """
    df = df.reset_index()
    sentence_vector = text_to_vector(sentences = df[col_to_vec], nlp = nlp)
    
    if feat_eng:
        num_feats = feature_eng(df, col_to_vec)
        x = pd.concat([df[num_feats], sentence_vector], axis=1)
    else:
        x = sentence_vector
        
    if target is not None:
        y = df[target]
    else: 
        y = None
    
    return x, y


###########################################
def df_preprocess(df, text_col, nlp_method = 'pre_trained_glove', target = None,  
                feat_eng = True, train = True , no_features = 1000, nlp = None):
    

    if nlp_method == 'pre_trained_glove':
        x , y = glove_preprocessing(df=df, col_to_vec=text_col,nlp = nlp, target=target, feat_eng=feat_eng)        
        
        
    elif nlp_method == 'tfidf':
        x, y = bow_preprocessing(df, text_col, target, feat_eng = feat_eng, train = train, no_features = no_features)

    
    else:
        raise ValueError("Invalid method. Implemented methods: 'pre_trained_glove', 'tfidf' ")
    return x , y



if __name__ == '__main__':
    pass
    
