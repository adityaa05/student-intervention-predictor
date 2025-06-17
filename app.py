from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import joblib

app = Flask(__name__)

# ✅ Load full model bundle
model_bundle = joblib.load("student_intervention_model_v3.pkl")
model = model_bundle["model"]
feature_cols = model_bundle["features"]
treatment_map = model_bundle["labels"]

# ✅ Load dataset for comparison (ensure this file is correct & preprocessed like your training data)
df = pd.read_csv("student_combined.csv")
if df["famsup"].dtype == object:
    df["famsup"] = df["famsup"].map({"yes": 1, "no": 0})

# ✅ Z-score heatmap generator
def generate_zscore_heatmap(student_dict, df):
    df_filtered = df[student_dict.keys()]
    student_df = pd.DataFrame([student_dict])

    z_scores = (student_df - df_filtered.mean()) / df_filtered.std()

    plt.figure(figsize=(10, 1.5))
    sns.heatmap(z_scores, annot=True, cmap="coolwarm", center=0, cbar=False)
    plt.title("Z-Score Heatmap (Student vs Class Avg)")
    plt.yticks(rotation=0)

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return plot_data

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "studytime": int(request.form["studytime"]),
        "goout": int(request.form["goout"]),
        "failures": int(request.form["failures"]),
        "famsup": int(request.form["famsup"]),
        "Dalc": int(request.form["Dalc"]),
        "Walc": int(request.form["Walc"]),
        "absences": int(request.form["absences"]),
        "G2": int(request.form["G2"])
    }

    features = np.array([[input_data[col] for col in feature_cols]])
    predicted_class = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label = treatment_map[predicted_class]
    probability_score = round(probabilities[list(model.classes_).index(predicted_class)] * 100, 2)

    class_probs = [
        (treatment_map[i], round(probabilities[idx] * 100, 2))
        for idx, i in enumerate(model.classes_)
    ]

    # ✅ Generate Z-score heatmap image
    heatmap_img = generate_zscore_heatmap(input_data, df)

    return render_template(
        "index.html",
        prediction=label,
        score=probability_score,
        probs=class_probs,
        heatmap=heatmap_img
    )

# ✅ START THE FLASK SERVER
if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    app.run(debug=True)
