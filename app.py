from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Save the trained model
joblib.dump(model, "model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([data["CGPA"], data["Internships"], data["Projects"], data["Workshops"], 
                         data["AptitudeTestScore"], data["SoftSkillsRating"], 
                         data["ExtracurricularActivities"], data["PlacementTraining"], 
                         data["SSC_Marks"], data["HSC_Marks"]]).reshape(1, -1)
    prediction = model.predict(features)
    result = "Placed" if prediction[0] == 1 else "Not Placed"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)