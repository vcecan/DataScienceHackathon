from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=["GET", "POST"])
def questions():
    if request.method == 'POST':
        nationality = request.form["nationality"]
        age = request.form["age"]
        why = request.form["why"]
        interest = request.form["interest"]
        marital = request.form["marital"]

        vector = [nationality, age, why, interest, marital]

        return vector

    return render_template('questions.html')

@app.route('/templates/notfound.html', methods=["GET", "POST"])
def notfound():
    return render_template('notfound.html')

@app.route('/templates/maps.html', methods=["GET", "POST"])
def maps():
    return render_template('maps.html')    

@app.route('/result', methods=["GET", "POST"])
def result():
    time.sleep(3)
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)