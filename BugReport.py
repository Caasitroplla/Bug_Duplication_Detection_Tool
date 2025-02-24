import pandas as pd
import numpy as np
import json
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec

from keras.layers import Input #type: ignore


class BugReport:

    word2vec_model: Word2Vec = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    vocab_initialized: bool = False

    def __init__(self, data_row: pd.Series) -> None:
        self.issue_id: int = int(data_row['Issue_id'].iloc[0])
        self.priority: str = str(data_row['Priority'].iloc[0])
        self.component: str = str(data_row['Component'].iloc[0])
        # If duplicated issue is NaN, set it to 0
        self.duplicated_issue: int = int(data_row['Duplicated_issue'].iloc[0]) if pd.notna(data_row['Duplicated_issue'].iloc[0]) else 0
        self.title: str = str(data_row['Title'].iloc[0])
        self.description: str = str(data_row['Description'].iloc[0])
        self.resolution: str = str(data_row['Resolution'].iloc[0])
        self.version: str = str(data_row['Version'].iloc[0])
        self.processed: np.ndarray | None = None

    @classmethod
    def from_save_file(cls, issue_id: int, processed: list[float], duplicated_issue: int) -> 'BugReport':
        self = cls.__new__(cls)
        self.issue_id = int(issue_id)
        self.processed = np.array(processed, dtype=np.float32)
        self.duplicated_issue = int(duplicated_issue)
        return self

    def process(self) -> None:
        concatenated_words: str = self.title + " " + self.description + " " + self.resolution
        self.processed = BugReport.preprocess_text(concatenated_words)

    def save_processed(self) -> None:
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
    def preprocess_text(text: str) -> np.ndarray:
        words: list[str] = text.split()

        # Remove empty strings, stop words, punctuation and numbers
        words = [word.lower() for word in words if word not in stopwords.words('english') and word not in string.punctuation and not word.isdigit()]

        # Lemmatize the words
        lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words] 

        # Initialize vocabulary if not already done
        if not BugReport.vocab_initialized:
            BugReport.word2vec_model.build_vocab([words])
            BugReport.vocab_initialized = True
        else:
            # Update the Word2Vec model with the new words
            BugReport.word2vec_model.build_vocab([words], update=True)

        # Get embeddings for each word and average them
        word_vectors: list[np.ndarray] = [BugReport.word2vec_model.wv[word] for word in words if word in BugReport.word2vec_model.wv]
        if word_vectors:
            processed: np.ndarray = np.mean(word_vectors, axis=0)
        else:
            processed: np.ndarray =  np.zeros(BugReport.word2vec_model.vector_size) # Assuming 100 dimensions for the embeddings

        return np.array([processed])


def get_processed_data() -> list[BugReport]:
    # Load the processed data from a file with the issue id, processed data and the duplicate issue id
    processed_data: list[BugReport] = []
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
