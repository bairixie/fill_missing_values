import os
import torch
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from transformers import BartTokenizer, BartForConditionalGeneration
from io import BytesIO
import csv
import logging

# Configure logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

MODEL_DIR = "./model/trained_bart_model"
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory {MODEL_DIR} not found. Ensure it is unzipped.")

tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval()

# Index route
@app.route('/')
def index():
    return render_template('web.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400

    try:
        # Detect and load the file with the correct delimiter
        delimiter = detect_delimiter(file)
        data = pd.read_csv(file, encoding='ISO-8859-1', sep=delimiter)
        logger.debug(f"File loaded successfully with shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return jsonify({"error": f"Failed to process file: {e}"}), 400

    try:
        # Predict missing values
        data_with_predictions = predict_missing_values(data)

        # Prepare previews
        original_preview = data.head(10).to_json(orient='split')
        predictions_preview = data_with_predictions.head(10).to_json(orient='split')

        # Save results for download
        output = BytesIO()
        data_with_predictions.to_csv(output, index=False)
        output.seek(0)
        upload_file.last_output_file = output

        return jsonify({
            "message": "File processed successfully",
            "original_preview": original_preview,
            "predictions_preview": predictions_preview,
            "download_url": request.host_url + "download_predictions"
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

# Download predictions
@app.route('/download_predictions', methods=['GET'])
def download_predictions():
    if not hasattr(upload_file, "last_output_file"):
        return jsonify({"error": "No file available for download"}), 400

    output = upload_file.last_output_file
    output.seek(0)
    return send_file(output, as_attachment=True, download_name='predicted_dataset.csv', mimetype='text/csv')

# Helper function: Detect delimiter
def detect_delimiter(file):
    try:
        sample = file.read(1024)
        file.seek(0)
        sniffer = csv.Sniffer()
        return sniffer.sniff(sample).delimiter
    except Exception as e:
        logger.warning(f"Failed to detect delimiter: {e}")
        return ','

# Predict missing values using the fine-tuned BART model
def predict_missing_values(data):
    data = data.copy()
    logger.debug("Starting missing value prediction")

    missing_indices = data.isnull()
    for col in ["Title", "Author", "Content"]:
        if col in data.columns and missing_indices[col].any():
            missing_rows = data.loc[missing_indices[col], :]
            try:
                logger.debug(f"Processing missing values for column: {col}")

                input_texts = missing_rows.fillna("").apply(
                    lambda row: f"Predict {col}: {row.drop(col).to_json()}", axis=1
                ).tolist()

                inputs = tokenizer(
                    input_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                logger.debug(f"Generated {len(inputs['input_ids'])} input sequences for column: {col}")

                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=100
                    )

                predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                data.loc[missing_indices[col], col] = predictions

                logger.debug(f"Filled {len(predictions)} missing values for column: {col}")
            except Exception as e:
                logger.error(f"Error predicting missing values for column {col}: {e}")
                data[col] = data[col].fillna("<PredictionError>")
    logger.debug("Missing value prediction completed")
    return data

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)