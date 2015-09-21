# -*- coding: utf-8 -*-
# Contact nuno.carneiro.farfetch.com for errors or if explanations are needed.

from flask import Flask, jsonify, abort, request, make_response
from datatransformation import COLUMNS_INPUT, COLUMNS_INDEX
from predictfraud import predictFraud

app = Flask('fraud')

@app.route('/predict', methods=['POST'])
def create_task():
    if not request.json:
        abort(400)

# COLUMNS_INDEX is just [0,1,...,n]. You'll need to change this variable it in datatransformation.py to the appropriate size.
# All type checks are done in datatransformation, in the function transformliverequest
    row = COLUMNS_INDEX
    for i in COLUMNS_INDEX:
        row[i] = request.json[COLUMNS_INPUT[i]]
    return str(predictFraud(row)), 201


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error 404': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=9999)