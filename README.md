# Bug_Duplication_Detection_Tool
Uses a CNN to detect duplicate bug reports

# Step 1: Data Collection & Preprocessing
## 1.1 Gather Bug Reports Dataset
Collect bug reports from platforms like Bugzilla, Jira, or open-source repositories.
Each bug report should have a title, description, and possibly metadata (e.g., timestamps, categories).
## 1.2 Preprocess Text Data
Tokenization: Convert sentences into tokens (words or subwords).
Stopword Removal: Remove unimportant words (e.g., "the," "is").
Stemming/Lemmatization: Normalize words (e.g., "running" → "run").
Word Embeddings: Convert words into numerical representations using:
Word2Vec, GloVe, or FastText
Pre-trained embeddings (e.g., BERT, RoBERTa)
## 1.3 Prepare Training Data
Label bug reports as duplicate (1) or non-duplicate (0).
Use cosine similarity or Jaccard similarity to find near-duplicate reports.
Format inputs as pairs of bug reports:
(Bug Report 1, Bug Report 2, Label)
# Step 2: Build a Convolutional Neural Network (CNN)
## 2.1 Define Input Layers
Use two input layers, one for each bug report in a pair.
Convert text into word embeddings (e.g., using pre-trained embeddings like Word2Vec).
## 2.2 CNN for Feature Extraction
Use 1D convolutional layers to extract text features.
Apply multiple convolutional filters with different kernel sizes to capture various n-gram features.
Use ReLU activation and max pooling for downsampling.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from tensorflow.keras.models import Model

# Define parameters
embedding_dim = 300  # Depends on pre-trained embeddings
max_length = 100     # Max number of words per bug report
vocab_size = 20000   # Adjust based on dataset size

# Input layers
input1 = Input(shape=(max_length,))
input2 = Input(shape=(max_length,))

# Embedding layer (using pre-trained embeddings)
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, trainable=False)

# CNN feature extractor
def cnn_block(input_layer):
    x = embedding_layer(input_layer)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    return x

# Extract features
features1 = cnn_block(input1)
features2 = cnn_block(input2)

# Concatenate extracted features
merged = Concatenate()([features1, features2])

# Fully connected layers
dense = Dense(128, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(dense)

# Build Model
model = Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

# Model summary
```python
model.summary()
```

# Step 3: Train the Model
## 3.1 Prepare Training & Test Data
Split the dataset into train (80%) and test (20%).
Convert words into integer sequences (using a tokenizer).
Apply padding to ensure uniform input length.
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize bug report texts
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(bug_reports)  # List of all bug reports

# Convert bug reports to sequences
sequences1 = tokenizer.texts_to_sequences(bug_reports1)
sequences2 = tokenizer.texts_to_sequences(bug_reports2)

# Pad sequences
X1 = pad_sequences(sequences1, maxlen=max_length, padding='post')
X2 = pad_sequences(sequences2, maxlen=max_length, padding='post')

# Labels
y = labels  # 1 if duplicate, 0 otherwise

# Train-test split
from sklearn.model_selection import train_test_split
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)
```

## 3.2 Train the CNN Model
```python
# Train model
model.fit([X1_train, X2_train], y_train, epochs=10, batch_size=32, validation_data=([X1_test, X2_test], y_test))
```

# Step 4: Evaluate & Improve
## 4.1 Evaluate Model Performance
```python
# Evaluate on test set
loss, accuracy = model.evaluate([X1_test, X2_test], y_test)
print(f'Test Accuracy: {accuracy:.4f}')
```

Use precision, recall, and F1-score to assess duplicate detection performance.
```python   
from sklearn.metrics import classification_report

y_pred = (model.predict([X1_test, X2_test]) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
```

## 4.2 Improve Performance
Use Bidirectional LSTM instead of CNN or a hybrid CNN + LSTM model.
Implement attention mechanisms for better context understanding.
Fine-tune pre-trained transformer models (BERT, RoBERTa).

# Step 5: Deploy the Model
## 5.1 Save the model:
```python
model.save("bug_duplicate_detector.h5")
```

## 5.2 Convert to TensorFlow Lite for mobile deployment.

## 5.3 Deploy via API using Flask or FastAPI.
