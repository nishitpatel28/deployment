from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask import Flask, request, g
from flask import send_file
from flask import render_template
from flask import jsonify
from PIL import Image, ExifTags
import tensorflow as tf
import numpy as np
import argparse

app = Flask(__name__)
model = None


def prepare_image(image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((64, 64))
    image = np.asarray(image)
    image = image.astype("float")/255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    image = request.files['file']
    image = Image.open(image)
    image = prepare_image(image)

    with graph.as_default():
        pred = model.predict(image)
        pred = pred.argmax(axis=1)[0]
    data['prediction'] = []
    if pred == 0:
        label = 'Parasitized'
    else:
        label = 'Uninfected'

    r = {'label': 'Sample', 'result': label}
    data['prediction'].append(r)
    data['success'] = True
    print(label + 'Image prediction Completed')
    return render_template('index.html', pred=label)


if __name__ == '__main__':
    print('Loading Model...')
    model = load_model('savedmodel.h5')
    graph = tf.get_default_graph()
    app.run(debug=True)
