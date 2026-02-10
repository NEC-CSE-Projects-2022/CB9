from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename
import joblib
import uuid
from datetime import datetime
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from preprocess import preprocess_inputs   # your custom function

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------------- Flask App Setup ----------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp_data')
MODEL_FOLDER = os.path.join(BASE_DIR, 'model')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- Load Model ----------------
def load_model():
    model_path = os.path.join(MODEL_FOLDER, "stacking_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("ERROR: 'stacking_model.pkl' missing inside /model folder!")

    model = joblib.load(model_path)
    print("🔥 Model Loaded Successfully")
    return model


model = load_model()


# ---------------- Prediction Logic ----------------
def make_prediction(filepath):
    try:
        # load file
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        # Check if dataframe is empty
        if df.empty:
            return None, None, "Error: Uploaded file is empty"

        # preprocess
        X = preprocess_inputs(df)

        # Check if preprocessing resulted in valid data
        if X.empty or X.isnull().all().all():
            return None, None, "Error: Could not process the file data"

        # model predict
        prob = model.predict_proba(X)[0][1]
        pred = int(prob >= 0.5)

        if pred == 1:
            result = f"🔥 High Chance of Forest Fire (Confidence: {prob:.4f})"
        else:
            result = f"🌿 Low Chance of Forest Fire (Confidence: {prob:.4f})"

        return result, f"{prob:.4f}", None

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Prediction Error: {error_msg}")
        print(traceback.format_exc())
        return None, None, f"Prediction Error: {error_msg}"


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about/about.html")


@app.route("/flowchart")
def flowchart():
    return render_template("flowchart/flowchart.html")


@app.route("/metrics")
def metrics():
    return render_template("metrics/metrics.html")


@app.route("/uploads", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("prediction/base.html")

    # Manual data
    if request.form.get("data_source") == "manual":
        timestamps = request.form.getlist('timestamp[]')
        action_types = request.form.getlist('action_type[]')
        item_ids = request.form.getlist('item_id[]')
        cursor_times = request.form.getlist('cursor_time[]')
        sources = request.form.getlist('source[]')
        answers = request.form.getlist('user_answer[]')
        platforms = request.form.getlist('platform[]')

        rows = []
        for i in range(len(timestamps)):
            rows.append({
                'timestamp': timestamps[i],
                'action_type': action_types[i],
                'item_id': item_ids[i],
                'cursor_time': cursor_times[i],
                'source': sources[i],
                'user_answer': answers[i],
                'platform': platforms[i]
            })

        df = pd.DataFrame(rows)
        filename = f"manual_{uuid.uuid4().hex[:6]}.csv"
        filepath = os.path.join(TEMP_FOLDER, filename)
        df.to_csv(filepath, index=False)

        return redirect(url_for("predict", filename=filename, source="manual"))

    # File Upload
    file = request.files.get("file")

    if not file or file.filename == "":
        flash("No file selected!", "danger")
        return redirect(url_for("upload_file"))

    if not allowed_file(file.filename):
        flash("Invalid file type!", "danger")
        return redirect(url_for("upload_file"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    return redirect(url_for("predict", filename=filename, source="file"))


@app.route("/predict")
def predict():
    filename = request.args.get("filename")
    source = request.args.get("source", "file")

    # If no filename provided, redirect to upload page
    if not filename:
        flash("Please upload a file first!", "info")
        return redirect(url_for("upload_file"))

    folder = TEMP_FOLDER if source == "manual" else UPLOAD_FOLDER
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        flash("File not found!", "danger")
        return redirect(url_for("upload_file"))

    result, probability, error = make_prediction(filepath)

    if error:
        flash(error, "danger")
        return redirect(url_for("upload_file"))

    df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath)
    
    return render_template(
        "prediction/results.html",
        upload_success=True,
        result=result,
        probability=probability,
        filename=filename,
        source=source,
        data_preview=df.head(10).to_dict("records"),
        data_columns=list(df.columns)
    )


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
