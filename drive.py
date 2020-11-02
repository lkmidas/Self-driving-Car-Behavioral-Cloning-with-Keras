import argparse
import base64
import cv2
from datetime import datetime
import os
import shutil
import time

import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from tensorflow.keras.backend import set_learning_phase
from tensorflow.keras.models import load_model


def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def preprocess(image, height, width):
    image = image[60:-25, :, :]
    image = cv2.resize(image, (width, height), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

sio = socketio.Server()
app = Flask(__name__)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MAX_SPEED = 30
AVG_SPEED = 20
MIN_SPEED = 10

speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        #steering_angle = float(data["steering_angle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        try:
            image = np.asarray(image)  # from PIL image to numpy array
            # Show image from central camera
            #image_bgr = cv2.GaussianBlur(image, (3, 3), 0)
            #image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #cv2.imshow("Camera", image_bgr)
            #cv2.waitKey(1)

            image = preprocess(image, 64, 64)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array
            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1))
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit, image_count, base_time
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED

            throttle = 1.0 - (steering_angle) ** 2 - (speed / speed_limit) ** 2

            #print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()

    #enable_xla()
    set_learning_phase(0)
    model = load_model(args.model)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

