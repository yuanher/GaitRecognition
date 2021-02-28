#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

from flask import Flask, flash, request, render_template, redirect, url_for, Response
import os

import io
import numpy as np
from shutil import copyfile
import json
import math

import tensorflow as tf

# Configure our application 
#

# Initialize our Flask app.
# NOTE: Flask is used to host our app on a web server, so that
# we can call its functions over HTTP/HTTPS.
#
app = Flask(__name__)

# Declare the labels we are using.
labels = ["S" + str(i).zfill(3) for i in range(1, 45)]
input = None

# function that computes length between two points
#   (x1,y1) - (x2,y2)
def compute_length(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

# Process Joints
# NOTE: The "x" parameter consists of an array of consecutive x and y values 
# within the same array.
def process_joints(x):

    r = [0] * 36

    # TODO:
    # Initialize some values for the reference length and the reference point.
    #
    #...#
    refx = 0
    refy = 0
    reflength = 1

    # TODO:
    # Step 1: Let's find the reference point (neck)
    #
    #...#
    if x[2] != 0 or x[3] != 0:         
        refx = x[2]                # use the neck X, Y
        refy = x[3]
    elif (x[4] != 0 or x[5] != 0) and (x[10] != 0 or x[11] != 0):
        refx = (x[4] + x[10]) / 2  # estimate the neck X, Y from the mid point
        refy = (x[5] + x[11]) / 2  # of the left/right shoulder
    
    # TODO:
    # Step 2: Let's first estimate the torso length.
    #
    #...#
    if x[16] != 0 and x[17] != 0:             
        reflength = compute_length(x[16], x[17], refx, refy)   # neck to right hip
    elif x[22] != 0 and x[23] != 0:
        reflength = compute_length(x[22], x[23], refx, refy)   # neck to left hip

    # TODO:
    # Step 3:
    # Perform the translation and the scaling.
    #
    #...#
    for i in range(0, 18):
        r[i*2] = (x[i*2] - refx) / reflength
        r[i*2 + 1] = (x[i*2 + 1] - refy) / reflength

    # Return the re-mapped and normalized result
    #
    return r

@app.route('/')
def upload_file():
   return render_template('file.html')

def retrieve_uploaded_file_name(request):
    f = request.files['file']
    filepath = "./assets/" + f.filename
    f.save(filepath)  
    return filepath
    
def preprocessKeypoints(name):
    file = "./assets/" + name + ".json"
    repetition_kps = []
    k = 3

    with open(file) as data_file: 
        data = json.load(data_file)
        for i, key in enumerate(data.keys()):
            no_of_frames = len(data[key]["people"][0]["pose_keypoints_2d"])
            pose_keypoints = data[key]["people"][0]["pose_keypoints_2d"]
            # remove every third element (confidence score) from pose_keypoints
            del pose_keypoints[k-1::k]
            repetition_kps.append(process_joints(pose_keypoints))

    print(np.array(repetition_kps).shape)
    return np.array(repetition_kps[:50]).reshape(1,50,36)

def generateKeypoints(fileName):
    # Call Alphapose
    print("Alphapose processing...")


@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded file names
    uploaded_file_name = retrieve_uploaded_file_name(request)     
    print("Uploaded File Name: " + uploaded_file_name)

    # Generate Keypoints from video file
    print("Generating Keypoints...")
    # python ../AlphaPose/scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --format open --video ./assets/S040_T2_L.avi --outdir ./assets
    generateKeypoints(uploaded_file_name)
    
    # Pre-process keypoints
    print("Preprocessing Keypoints...")
    name = uploaded_file_name.split("/")[-1].split(".")[0]
    input = preprocessKeypoints(name)

    print(input.shape)

    # Load Model
    print("Loading Model...")
    loaded_model = tf.keras.models.load_model('model')
 
    # Predict
    print("Generating prediction...")
    y = loaded_model.predict(input)

    # Get the best class index and corresponding text label and prediction score
    label_index = np.argmax(y)
    print("Label Index:" + str(label_index))

    label_score = round(y[0][label_index],3)
    print("Label Score:" + str(label_score))

    label_text = labels[label_index]
    print("Label Text:" + label_text)

    # Return prediction message
    print("Returning prediction message...")
    return render_template('file.html', prediction_text='Prediction: {}, Score: {}'.format(label_text, label_score))
    
#------------------------------------------------------------------------------
# This starts our web server.
# Although we are running this on our local machine,
# this can technically be hosted on any VM server in the cloud!
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=80)



