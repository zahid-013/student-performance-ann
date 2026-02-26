from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import pickle

from model_def import StudentPerformanceNN 

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
device = torch.device("cpu")

model = StudentPerformanceNN(input_size=19)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# OPTIONAL: load scaler if used
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    scaler = None

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        features = np.array([
            float(data["Hours_Studied"]),
            float(data["Attendance"]),
            int(data["Parental_Involvement"]),
            int(data["Access_to_Resources"]),
            int(data["Extracurricular_Activities"]),
            float(data["Sleep_Hours"]),
            float(data["Previous_Scores"]),
            int(data["Motivation_Level"]),
            int(data["Internet_Access"]),
            int(data["Tutoring_Sessions"]),
            int(data["Family_Income"]),
            int(data["Teacher_Quality"]),
            int(data["School_Type"]),
            int(data["Peer_Influence"]),
            float(data["Physical_Activity"]),
            int(data["Learning_Disabilities"]),
            int(data["Parental_Education_Level"]),
            float(data["Distance_from_Home"]),
            int(data["Gender"])
        ]).reshape(1, -1)

        # ✅ apply scaler if exists
        if scaler is not None:
            features = scaler.transform(features)

        tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            output = model(tensor)
            prediction = round(output.item(), 2)

        return jsonify({"prediction": float(prediction*100)})

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)