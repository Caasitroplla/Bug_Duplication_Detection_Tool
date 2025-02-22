# Bug_Duplication_Detection_Tool
Uses a CNN to detect duplicate bug reports

To use run the rest api, run the app.py script:

```bash
python app.py
```

This will start the rest api on port 5000.

To test the rest api, you can use the following curl command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"bug_report_1": "Bug report 1", "bug_report_2": "Bug report 2"}' http://localhost:5000/predict
```

Or alternatively, you can use the following python code:

```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "bug_report_1": "App crashes when I click save",
    "bug_report_2": "The application stops working when saving"
}

response = requests.post(url, json=data)
print(response.json())  # {"prediction": "Duplicate", "confidence": 0.87}
```

If you whish to retrain the model, run `main.py` this will ensure all current bug report have been processed which are then stored in `processed_data.json` before retraining the model of the data stored the file. This model is then saved as `model.keras` which is used by the rest api.