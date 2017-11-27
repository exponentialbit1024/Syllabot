from flask import Flask, jsonify, request, session, abort, render_template
from flask_cors import CORS
import json

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
        return jsonify({'result': True, 'response': "Got something"})

    return jsonify({'result': False, 'error': "empty input"})
