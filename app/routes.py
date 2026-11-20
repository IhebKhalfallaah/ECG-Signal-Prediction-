"""HTTP routes: landing page, login, patient form and prediction."""
import os

from flask import current_app, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from ecg.plotting import generate_plots
from ecg.predict import predict_majority_class_from_file

ALLOWED_EXTENSIONS = (".dat", ".hea", ".xyz", "atr")
PATIENT_FIELDS = ("name", "phone", "dob", "address", "weight", "height", "sexe", "symptomes")


def home():
    return render_template("home.html")


def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == current_app.config["USERNAME"] and password == current_app.config["PASSWORD"]:
            return redirect(url_for("predictPage"))
        error = "Invalid username or password"
    return render_template("login.html", error=error)


def predictPage():
    return render_template("predict.html")


def _save_uploads(upload_folder, files):
    """Replace the upload folder content with the accepted ECG files."""
    for f in os.listdir(upload_folder):
        os.remove(os.path.join(upload_folder, f))
    for file in files:
        if file and file.filename.endswith(ALLOWED_EXTENSIONS):
            filename = secure_filename(file.filename.split("/")[-1])
            file.save(os.path.join(upload_folder, filename))


def _has_required_files(upload_folder, record):
    exists = lambda ext: os.path.exists(os.path.join(upload_folder, f"{record}.{ext}"))
    return exists("dat") and exists("hea") and (exists("xyz") or exists("atr"))


def predict():
    patient = {field: request.form.get(field) for field in PATIENT_FIELDS}
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    _save_uploads(upload_folder, request.files.getlist("ecg_files"))

    base_records = {f.split(".")[0] for f in os.listdir(upload_folder)}
    if len(base_records) != 1:
        return render_template(
            "predict.html",
            error="❌ You must upload exactly one ECG record (.dat, .hea, and .xyz files for one patient)",
            **patient)

    record = next(iter(base_records))
    if not _has_required_files(upload_folder, record):
        return render_template("predict.html", error="❌ Missing .dat, .hea, or .xyz file", **patient)

    try:
        record_path = os.path.join(upload_folder, record)
        prediction = predict_majority_class_from_file(record_path, current_app.config["MODEL_PATH"])
        results = [{"record": record, "prediction": prediction}]
        plots = generate_plots(record_path, record, label=results, plots_dir=current_app.config["PLOTS_DIR"]) or {}
    except Exception as e:
        results = [{"record": record, "error": f"❌ Prediction failed: {str(e)}"}]
        plots = {}

    return render_template(
        "result.html",
        **patient,
        results=results,
        plots=plots,
        metrics_iii=plots.get("metrics_iii", {}),
        metrics_v4=plots.get("metrics_v4", {}))


def register_routes(app):
    app.add_url_rule("/", view_func=home, methods=["GET"])
    app.add_url_rule("/login", view_func=login, methods=["GET", "POST"])
    app.add_url_rule("/predictPage", view_func=predictPage, methods=["GET"])
    app.add_url_rule("/Predict", view_func=predict, methods=["POST"])
