# flask_app/app.py
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the model and resources during initialization
loaded_model = None
appreciate = None
vocabulary = None

@app.before_first_request
def load_model_and_resources():
    global loaded_model, appreciate, vocabulary
    loaded_model = load_model("nwp_model.h5")
    appreciate = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    vocabulary = np.load('vocabulary.npy')

# Create a function to predict the next word
def next_word(collection=[], extent=1):
    for item in collection:
        text = item
        for i in range(extent):
            prediction = loaded_model.predict(x=appreciate([item]).numpy())
            idx = np.argmax(prediction[-1])
            item += ' ' + vocabulary[idx]
        return item.split(' ')[-1]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_sequence():
    prompt = request.form.get('prompt')
    no_words = int(request.form.get('no_words'))
    sentence = prompt
    text_collection = prompt.split(" ")
    word = text_collection[-1]
    for i in range(no_words):
        next = next_word([word])
        sentence = sentence + " " + next
        word = next
    return jsonify({'generated_sequence': sentence, 'no_words': no_words})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True)
