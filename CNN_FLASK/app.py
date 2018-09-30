#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:36:42 2018

@author: aakanch
"""

import os, time, datetime
from flask import Flask, render_template, request
import sys
import argparse
import numpy as np
import pandas as pd
import PIL.Image
import requests
from io import BytesIO
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pygal
import tensorflow as tf

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)
model = load_model("model.h5")
graph = tf.get_default_graph()


@app.route("/", methods=['GET'])
def server():
    return render_template('index.html')

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/home", methods=['GET'])
def ret():
    return render_template('index.html')

def predict(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    with graph.as_default():
            preds = model.predict(x)
    return preds[0]

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    name, ext = file.filename.split(".")
    name_r = name
    date, Time = time.strftime("%x %X", time.localtime()).split(" ")
    date = date.replace("/","-")
    name = name + "_" + date + "_" + Time
    file.filename = name + "." + ext
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)

    """Here on code for prediction is entered """

    path = "uploads" + "/" + file.filename
    img = image.load_img(path)

    img = img.resize((64,64))

    prediction = predict(img)
    if prediction[0] > 0.1:
        pred_class = 'cat'
    else:
        pred_class = 'dog'

    """Code for exporting data"""

    return render_template('Report.html', result=pred_class)

if __name__ == '__main__':
    app.debug = True
    host = os.environ.get('IP', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    app.run(host=host, port=port)
