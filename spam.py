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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud

class CustomTokenizer(object):
    def __init__(self, tokenizer: TokenizerI, stemmer: StemmerI=None, lower_case: bool=True, exclude_stopwords: bool=True) -> None:
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.lower_case = lower_case
        self.exclude_stopwords = exclude_stopwords
    
    def tokenize(self, message: str) -> Counter:
        if self.lower_case:
            message = message.lower()
        
        words = self.tokenizer.tokenize(message)
    
        if self.exclude_stopwords:
            exclude_words = stopwords.words('english')
            words = [word for word in words if word not in exclude_words]
        
        if self.stemmer != None:
            words = [self.stemmer.stem(word) for word in words]
        
        bag = Counter()
        
        for word in words:
            if word not in bag:
                bag[word] = 1
            else:
                bag[word] += 1
        return bag

class CustomVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer: 'CustomTokenizer', num_ham_words: int=100, num_spam_words: int=100):
        super().__init__()
        self.num_ham_words = num_ham_words
        self.num_spam_words = num_spam_words
        self.tokenizer = tokenizer
        self.words = None
        self.ham_words = None
        self.spam_words = None
        
    def fit(self, X, y) -> np.array:
        spam_words = Counter()
        ham_words = Counter()
        
        for index, row in X.iteritems():
            for word in self.tokenizer.tokenize(row):
                if y[index] == 0:
                    if word not in ham_words:
                        ham_words[word] = 1
                    else:
                        ham_words[word] += 1
                elif y[index] == 1:
                    if word not in spam_words:
                        spam_words[word] = 1
                    else:
                        spam_words[word] += 1
                else:
                    raise ValueError('y can only take on 0 (ham) or 1 (spam). Got {}.'.format(y[index]))
        
        ham_word_freqs = sorted(ham_words.items(), key=lambda x: x[1], reverse=True)
        spam_word_freqs = sorted(spam_words.items(), key=lambda x: x[1], reverse=True)
        
        ham_words = [word[0] for word in ham_word_freqs]
        spam_words = [word[0] for word in spam_word_freqs]
        
        if len(ham_words) >= self.num_ham_words:
            self.ham_words = ham_words[:self.num_ham_words]
        else:
            self.ham_words = ham_words
        
        if len(spam_words) >= self.num_spam_words:
            self.spam_words = spam_words[:self.num_spam_words]
        else:
            self.spam_words = spam_words
        
        self.words = self.ham_words + spam_words
        return self
    
    def transform(self, X) -> dok_matrix:
        if self.words == None:
            raise Exception('Fit must be run before transform.')
        
        features = dok_matrix((len(X), len(self.words)), dtype=np.float32)
        for i in range(len(X)):
            message = self.tokenizer.tokenize(X.iloc[i])
            for j in range(len(self.words)):
                if self.words[j] in message:
                    features[i, j] = message[self.words[j]]
        return features.toarray()
    
    def words_to_json(self, filename: str) -> None:
        if self.words == None:
            raise Exception('No words extracted. Run the extract method.')
            
        with open(filename, 'w') as file:
            json.dump(self.words, file, indent=4)

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
    print('The total ham/spam ratio is {}.'.format(original_spam_ratio))
    
    X = spam_dataset['message']
    y = spam_dataset['spam']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y)
    
    train_spam_ratio = len(y_train[y_train == 1])/len(y_train)
    print('The training set ham/spam ratio is {}.'.format(train_spam_ratio))
    
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
    
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    custom_tokenizer = CustomTokenizer(tokenizer, stemmer)
    
    word_analyzer = CustomVectorizer(custom_tokenizer, num_ham_words=100, num_spam_words=100)
    n_bayes = MultinomialNB(alpha=1.0)
    
    pipe = Pipeline([
        ('word_analyzer', word_analyzer),
        ('naive_bayes', n_bayes),
        ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    pipe.score(X_test, y_pred)
    
    print('The accuracy score is {}.'.format(accuracy_score(y_test, y_pred)))
    print('The precision score is {}.'.format(precision_score(y_test, y_pred)))
    print('The recall score is {}.'.format(recall_score(y_test, y_pred)))
    print('The f1 score is {}.'.format(f1_score(y_test, y_pred)))
    
    print('The confusion matrix is:')
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()