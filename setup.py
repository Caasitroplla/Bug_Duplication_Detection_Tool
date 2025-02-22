import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

import rarfile

rar = rarfile.RarFile('eclipse_platform.rar')
rar.extractall(path='data')