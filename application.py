from flask import Flask, render_template
application = Flask(__name__)

@application.route("/index")
def index():
    return render_template("index.html")
