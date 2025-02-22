import pandas as pd
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
from gensim.models import Word2Vec
from keras.layers import Input #type: ignore


class BugReport:

    word2vec_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    vocab_initialized = False

    def __init__(self, data_row):
        self.issue_id = int(data_row['Issue_id'].iloc[0])
        self.priority = str(data_row['Priority'].iloc[0])
        self.component = str(data_row['Component'].iloc[0])
        # If duplicated issue is NaN, set it to 0
        self.duplicated_issue = int(data_row['Duplicated_issue'].iloc[0]) if pd.notna(data_row['Duplicated_issue'].iloc[0]) else 0
        self.title = str(data_row['Title'].iloc[0])
        self.description = str(data_row['Description'].iloc[0])
        self.resolution = str(data_row['Resolution'].iloc[0])
        self.version = str(data_row['Version'].iloc[0])
        self.processed = None

    @classmethod
    def from_save_file(cls, issue_id, processed, duplicated_issue):
        self = cls.__new__(cls)
        self.issue_id = int(issue_id)
        self.processed = np.array(processed, dtype=np.float32)
        self.duplicated_issue = int(duplicated_issue)
        return self

    def process(self):
        # From all string data types, split them into words
        words = []
        words.extend(word_tokenize(self.title))
        words.extend(word_tokenize(self.description))
        words.extend(word_tokenize(self.resolution))

        # Remove empty strings, stop words, punctuation and numbers
        words = [word.lower() for word in words if word not in stopwords.words('english') and word not in string.punctuation and not word.isdigit()]

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Remove duplicates
        words = list(set(words))

        # Initialize vocabulary if not already done
        if not BugReport.vocab_initialized:
            BugReport.word2vec_model.build_vocab([words])
            BugReport.vocab_initialized = True
        else:
            # Update the Word2Vec model with the new words
            BugReport.word2vec_model.build_vocab([words], update=True)

        # Train the model with the new words
        BugReport.word2vec_model.train([words], total_examples=1, epochs=1)

        # Get embeddings for each word and average them
        word_vectors = [BugReport.word2vec_model.wv[word] for word in words if word in BugReport.word2vec_model.wv]
        if word_vectors:
            self.processed = np.mean(word_vectors, axis=0)
        else:
            self.processed = np.zeros(BugReport.word2vec_model.vector_size)  # Assuming 100 dimensions for the embeddings

    

    def save_processed(self):
        
        # Save the processed data to a file with the issue id, processed data and the duplicate issue id
        with open('processed_data.json', 'a') as f:
            json.dump({
                'issue_id': self.issue_id,
                'processed': self.processed.tolist(),
                'duplicated_issue': self.duplicated_issue
            }, f)
            f.write('\n')

    # Copy of the process method but for a single piece of text
    @staticmethod
    def preprocess_text(text):
        words = text.split()

        # Remove empty strings, stop words, punctuation and numbers
        words = [word.lower() for word in words if word not in stopwords.words('english') and word not in string.punctuation and not word.isdigit()]

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words] 

        # Initialize vocabulary if not already done
        if not BugReport.vocab_initialized:
            BugReport.word2vec_model.build_vocab([words])
            BugReport.vocab_initialized = True
        else:
            # Update the Word2Vec model with the new words
            BugReport.word2vec_model.build_vocab([words], update=True)

        # Get embeddings for each word and average them
        word_vectors = [BugReport.word2vec_model.wv[word] for word in words if word in BugReport.word2vec_model.wv]
        if word_vectors:
            processed = np.mean(word_vectors, axis=0)
        else:
            processed =  np.zeros(BugReport.word2vec_model.vector_size)

        return np.array([processed])


        
        

def get_processed_data():
    # Load the processed data from a file with the issue id, processed data and the duplicate issue id
    processed_data = []
    with open('processed_data.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            processed_data.append(BugReport.from_save_file(data['issue_id'], data['processed'], data['duplicated_issue']))

    return processed_data


# For testing purposes
# if __name__ == "__main__":

#     # Load the data
#     data = pd.read_csv('data/eclipse_platform.csv', encoding='utf-8')

#     data.columns = data.columns.str.strip()

#     # To test load the first 3 rows
#     data = data.head(3)

#     bug_reports = []

#     # Process the data
#     for index, row in data.iterrows():
#         bug_report = BugReport(row)
#         bug_report.process()
#         print(bug_report.processed.shape)
#         print(bug_report.processed)
#         bug_reports.append(bug_report)

#     # Blank the file
#     with open('processed_data.json', 'w') as f:
#         f.truncate(0)

#     for bug_report in bug_reports:
#         bug_report.save_processed()

#     processed_data = get_processed_data()

#     for bug_report in processed_data:
#         print(bug_report.processed)
