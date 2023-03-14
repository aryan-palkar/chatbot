import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from flask_cors import CORS

import random
import pickle

with open("intents1.json") as file:
    data = json.load(file)


model = keras.models.load_model('chat_model')

# load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# parameters
max_len = 20

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def chat():
    dta = request.get_json()
    inp = dta['msg']
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    for i in data['intents']:
        if i['tag'] == tag:
            # print()

            res = {"bot" : np.random.choice(i['responses'])}
            return jsonify(res)

if __name__ == '__main__':
    app.run(debug = True)

