from flask import Flask, jsonify, request, session, abort, render_template
from flask_cors import CORS
import json
from .dbOb import *
from .dataCon import *

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
        dbObj = dbOb()
        dconOb = dataCon(dbObj)
        saveSuc = dconOb.save_text(inputText)
        if saveSuc:
            return jsonify({'result': True, 'response': "Got something"})
        return jsonify({'result' : False, 'error' : "Couldn\'t save"})

    return jsonify({'result': False, 'error': "empty input"})
