from flask import Flask
from flask import render_template
from poetry_agent import PoetryAgent


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('/home.html')

@app.route("/poems")
def poems():
    return render_template('/poems.html')

