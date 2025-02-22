from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

from BugReport import BugReport

# Load the model
model = tf.keras.models.load_model('model.keras')

app = Flask(__name__)

def preprocess_text(text):
    return BugReport.preprocess_text(text)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json

        # Extract bug report texts
        text1 = data.get("bug_report_1", "")
        text2 = data.get("bug_report_2", "")

        # Preprocess the texts
        processed_text1 = preprocess_text(text1)
        processed_text2 = preprocess_text(text2)

        # Make the prediction
        prediction = model.predict([processed_text1, processed_text2])

        # Convert the prediction to human readable format
        result = "Duplicate" if prediction > 0.5 else "Not Duplicate"

        return jsonify({"prediction": result, "confidence": float(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
