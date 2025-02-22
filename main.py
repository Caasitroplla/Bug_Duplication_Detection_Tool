import pandas as pd
from BugReport import BugReport
import json
import random
import numpy as np

import tensorflow as tf
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split

# Read the data from the csv file
def read_csv_file(file_path = "data/eclipse_platform.csv"):
    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    return df

# Process the data into a format ready for training
def process_data(df):
    # Look for the highest issue id in the allready processed json file (will be bottom entry)
    with open('processed_data.json', 'r') as f:
        # Get the last line
        last_line = f.readlines()[-1]
        # Load the json
        last_line = json.loads(last_line)
        # Get the issue id
        current_issue_id = last_line['issue_id']


    highest_issue_id = df['Issue_id'].max()

    if current_issue_id >= highest_issue_id:
        print("All issues already processed")
        return
    
    # Process the data from the highest issue id to the highest issue id in the dataframe
    for issue_id in range(current_issue_id + 1, highest_issue_id + 1):
        # Get the row
        row = df[df['Issue_id'] == issue_id]

        # Make sure the row is not empty
        if row.empty:
            print(f"Issue id {issue_id} is empty")
            continue

        # Create a bug report
        bug_report = BugReport(row)
        # Process the bug report
        bug_report.process()
        # Save the bug report by appending it to the json file array
        with open('processed_data.json', 'a') as f:
            json.dump({
                'issue_id': bug_report.issue_id,
                'processed': bug_report.processed.tolist(),
                'duplicated_issue': bug_report.duplicated_issue
            }, f)
            f.write('\n')

def load_processed_data():
    bug_reports = []

    with open('processed_data.json', 'r') as f:
        data = f.readlines()
        # For each line, create a bug report object
        for line in data:
            data = json.loads(line)
            bug_reports.append(BugReport.from_save_file(data['issue_id'], data['processed'], data['duplicated_issue']))

    return bug_reports

def create_pairs(bug_reports):
    # Create pairs of bug reports and equal amount of duplicated and non duplicates
    duplicated_bug_reports = [bug_report for bug_report in bug_reports if bug_report.duplicated_issue != 0]

    # Create pairs of duplicated bug reports
    duplicated_pairs = []
    # The value of 'duplicated_issue' contains then issue id of the duplicated bug report
    for duplicated_bug_report in duplicated_bug_reports:
        # Get the duplicated issue id
        duplicated_issue_id = duplicated_bug_report.duplicated_issue
        # Get the duplicated bug report - may not exsists if so skip
        try:
            duplicated_bug_report = bug_reports[duplicated_issue_id]
        except:
            continue

        # Create a pair of the duplicated bug report and the duplicated bug report and add the result of 1 to the pair
        duplicated_pairs.append((duplicated_bug_report.processed, duplicated_bug_report.processed, 1))

    # Create an equal amount of pairs of non duplicated bug reports
    non_duplicated_pairs = []
    ammount_of_pairs = len(duplicated_pairs)

    # Create pairs of non duplicated bug reports
    for _ in range(ammount_of_pairs):
        # Create a pair of the non duplicated bug report and the random non duplicated bug report and add the result of 0 to the pair
        non_duplicated_pairs.append((random.choice(bug_reports).processed, random.choice(bug_reports).processed, 0))

    # Randomly shuffle the two pairs together
    pairs = duplicated_pairs + non_duplicated_pairs
    random.shuffle(pairs)
    return pairs

# # Read the data
# df = read_csv_file()

# print(df.head())
# # Process the data
# process_data(df)

bug_reports = load_processed_data()

pairs = create_pairs(bug_reports)

# Assuming pairs is a list of tuples: (vector1, vector2, label)
# Example: pairs = [(vec1, vec2, 1), (vec3, vec4, 0), ...]

# Prepare the data
X1 = np.array([pair[0] for pair in pairs])  # First bug report vectors
X2 = np.array([pair[1] for pair in pairs])  # Second bug report vectors
y = np.array([pair[2] for pair in pairs])   # Labels

# Ensure X1 and X2 are 2D arrays
if X1.ndim == 1:
    X1 = X1.reshape(-1, 1)
if X2.ndim == 1:
    X2 = X2.reshape(-1, 1)

# Split the data into training and testing sets
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# Define the model
input1 = Input(shape=(X1.shape[1],))
input2 = Input(shape=(X2.shape[1],))

# Shared dense layer
shared_dense = Dense(128, activation='relu')

# Process both inputs through the shared layer
x1 = shared_dense(input1)
x2 = shared_dense(input2)

# Concatenate the processed inputs
concatenated = Concatenate()([x1, x2])

# Add a dense layer and output layer
x = Dense(64, activation='relu')(concatenated)
output = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X1_train, X2_train], y_train, epochs=10, batch_size=32, validation_data=([X1_test, X2_test], y_test))

# Save the model
model.save('model.keras')

loss, accuracy = model.evaluate([X1_test, X2_test], y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")