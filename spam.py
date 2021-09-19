# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:02:06 2021

@author: kvonk
"""

from collections import Counter
import json
import matplotlib.pyplot as plt
from nltk.tokenize.api import TokenizerI
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.api import StemmerI
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from typing import List
from wordcloud import WordCloud

class FeatureModel(object):
    def __init__(self, X, y, num_ham_words: int=100, num_spam_words: int=100):
        self.num_ham_words = num_ham_words
        self.num_spam_words = num_spam_words
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = PorterStemmer()
        self.X = X
        self.y = y
        self.messages = []
        self.spam_words = Counter()
        self.ham_words = Counter()
        self.all_words = Counter()
        
        for index, row in X.iteritems():
            message = Counter()
            
            for word in process_message(row, self.tokenizer, self.stemmer):

                if word not in self.all_words:
                    self.all_words[word] = 1
                else:
                    self.all_words[word] += 1
                
                if y[index] == 0:
                    if word not in self.ham_words:
                        self.ham_words[word] = 1
                    else:
                        self.ham_words[word] += 1
                
                if y[index] == 1:
                    if word not in self.spam_words:
                        self.spam_words[word] = 1
                    else:
                        self.spam_words[word] += 1
                
                if word not in message:
                    message[word] = 1
                else:
                    message[word] += 1
            
            self.messages.append(message)
        
        ham_list = sorted(self.ham_words.items(), key=lambda x: x[1], reverse=True)
        spam_list = sorted(self.spam_words.items(), key=lambda x: x[1], reverse=True)

        word_set = set()
        
        for i in range(num_ham_words):
            word_set.add(ham_list[i][0])
        
        for i in range(num_spam_words):
            word_set.add(spam_list[i][0])
        
        self.word_list = list(word_set)
    
def process_message(message: str, tokenizer: TokenizerI, stemmer: StemmerI=None, lower_case: bool=True, exclude_stopwords: bool=True) -> List[str]:
    if lower_case:
        message = message.lower()
    
    words = tokenizer.tokenize(message)

    if exclude_stopwords:
        exclude_words = stopwords.words('english')
        words = [word for word in words if word not in exclude_words]
    
    if stemmer != None:
        words = [stemmer.stem(word) for word in words]
    
    bag = Counter()
    
    for word in words:
        if word not in bag:
            bag[word] = 1
        else:
            bag[word] += 1

    return bag
    
def get_features(X, feature_model: 'FeatureModel') -> dok_matrix:
    features = dok_matrix((len(X), len(feature_model.word_list)), dtype=np.float32)
    for i in range(len(X)):
        message = process_message(X.iloc[i], feature_model.tokenizer, feature_model.stemmer)
        for j in range(len(feature_model.word_list)):
            if feature_model.word_list[j] not in message:
                features[i, j] = 0
            else:
                features[i, j] = message[feature_model.word_list[j]]
    return features.toarray()



def main():
    spam_dataset = pd.read_csv(
    'spam.csv',
    encoding='ISO-8859-1',
    usecols=[
        0,
        1,
    ],
    names=[
        'spam',
        'message',
    ],
    header=0)

    spam_dataset['spam'] = np.where(spam_dataset['spam'].str.contains('spam'), 1, 0)
    
    original_spam_ratio = len(spam_dataset[spam_dataset['spam'] == 1])/len(spam_dataset['spam'])
    print(original_spam_ratio)
    
    X = spam_dataset['message']
    y = spam_dataset['spam']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y)
    
    train_spam_ratio = len(y_train[y_train == 1])/len(y_train)
    print(train_spam_ratio)
    
    spam_words = ' '.join(list(X_train[y_train == 1]))
    spam_wc = WordCloud(width=512, height=512).generate(spam_words)
    plt.figure(figsize=(10, 8), facecolor='k')
    plt.imshow(spam_wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
    ham_words = ' '.join(list(X_train[y_train == 0]))
    ham_wc = WordCloud(width=512, height=512).generate(ham_words)
    plt.figure(figsize=(10, 8), facecolor='k')
    plt.imshow(ham_wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
    feature_model = FeatureModel(X_train, y_train)

    features_train = get_features(X_train, feature_model)
    
    cnb = MultinomialNB(alpha=1.0)
    
    cnb.fit(features_train, y_train)
    
    cnb.predict(features_train[1:20])
    
    features_test = get_features(X_test, feature_model)
    
    y_pred = cnb.predict(features_test)
    cnb.score(features_test, y_test)
    
    print('The accuracy score is {}.'.format(accuracy_score(y_test, y_pred)))
    print('The precision score is {}.'.format(precision_score(y_test, y_pred)))
    print('The recall score is {}.'.format(recall_score(y_test, y_pred)))
    print('The f1 score is {}.'.format(f1_score(y_test, y_pred)))
    
    print('The confusion matrix is:')
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()