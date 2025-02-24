import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify

from BugReport import BugReport

# Load the model
model: tf.keras.Model = tf.keras.models.load_model('model.keras')

app: Flask = Flask(__name__)

def preprocess_text(text: str) -> np.ndarray:
    return BugReport.preprocess_text(text)

@app.route('/predict', methods=['POST'])
def predict() -> tuple[dict[str, str], int]:
    try:
        # Get the input data from the request
        data: dict[str, str] = request.json

        # Extract bug report texts
        text1: str = data.get("bug_report_1", "")
        text2: str = data.get("bug_report_2", "")

        # Preprocess the texts
        processed_text1: np.ndarray = preprocess_text(text1)
        processed_text2: np.ndarray = preprocess_text(text2)

        # Make the prediction
        prediction: np.ndarray = model.predict([processed_text1, processed_text2])

        # Convert the prediction to human readable format
        result: str = "Duplicate" if prediction > 0.5 else "Not Duplicate"

        return jsonify({"prediction": result, "confidence": float(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
