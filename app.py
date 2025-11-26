from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("models/best_xgbr_model_final.pkl", "rb"))

FEATURE_COLUMNS = [
    "Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp",
    "Duration_Heart", "Duration_Temp", "Age_Duration", "Weight_Duration",
    "Height_Duration", "HR_per_Weight"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    form_data = request.form.to_dict()

    input_df = pd.DataFrame([form_data])

    numeric_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    input_df["Sex"] = input_df["Sex"].map({"male": 0, "female": 1}).astype(float)
    input_df["Duration_Heart"]   = input_df["Duration"] * input_df["Heart_Rate"]
    input_df["Duration_Temp"]    = input_df["Duration"] * input_df["Body_Temp"]
    input_df["Age_Duration"]     = input_df["Age"]      * input_df["Duration"]
    input_df["Weight_Duration"]  = input_df["Weight"]   * input_df["Duration"]
    input_df["Height_Duration"]  = input_df["Height"]   * input_df["Duration"]
    input_df["HR_per_Weight"]    = input_df["Heart_Rate"] / input_df["Weight"].replace(0, np.nan)

    input_df = input_df.fillna(0)
    input_df = input_df[FEATURE_COLUMNS]

    pred = model.predict(input_df)[0]
    pred = float(np.clip(pred, 0, None))

    return render_template("result.html", prediction=round(pred, 2))

if __name__ == "__main__":
    app.run(debug=True)

