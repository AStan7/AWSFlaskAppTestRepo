from flask import Flask
application = Flask(__name__)

@application.route("/")
def hello():
    return "Is this working?"