'''This file is used to download the necessary data for nltk to work and extract the data from the rar file used to train the model'''

import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

import rarfile

rar = rarfile.RarFile('eclipse_platform.rar')
rar.extractall(path='data')