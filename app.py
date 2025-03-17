from flask import Flask, render_template, request
import pickle

# Load the trained model
with open("spam_detector.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    prediction = model.predict([message])[0]
    return render_template("result.html", message=message, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)