from flask import Flask, jsonify, request, session, abort, render_template
from flask_cors import CORS
import json
from .dbOb import *
from .dataCon import *
from .classify_online import *

app = Flask(__name__)
app.secret_key = "M0vR7JyggCjJBM3nNAFhJfgQuPxym46wFuUbTeA3"
CORS(app)

@app.route("/")
def testBackend():
    return "server works"

@app.route("/textIn", methods = ["POST"])
def predictChatOut():
    if request.method != "POST":
        return jsonify({'result' : False, 'error' : "Improper request type"})

    payload = json.loads(request.data.decode())
    inputText = payload['input']
    if len(inputText) > 0:
        print(inputText)
        if inputText.lower() == "hi" or inputText.lower() == "hello":
            return jsonify({'result' : "True", 'response' : "Hello! I\'m Syllabot! You can ask me general questions about the course like, \"When is the final?\" "})
        clf = classifier()
        dbObj = dbOb()
        dconOb = dataCon(dbObj)
        saveSuc = dconOb.save_text(inputText)
        prediction = clf.predict(inputText)
        return jsonify({'result' : True, 'response' : prediction})

    return jsonify({'result': False, 'error': "empty input"})
