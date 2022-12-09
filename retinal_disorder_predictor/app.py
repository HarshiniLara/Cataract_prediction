import os
import flask
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import img_to_array
import pyttsx3



app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR , 'resnet_model.hdf5'))

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


image_size = 224
prevention = {
    "normal": "You are healthy.",
    "cataract": "You are affected by Cataract. Have a consistently healthy diet that includes fruits, vegetables, oily fish and whole grains. Stay away from UV radiation. Stop smoking. Control your blood sugar. Avoid the unnecessary use of steroids. Sunglasses can also help cut your risk of getting cataracts. Get regular eye checkup"
}

def predictResult(filename , model):
    img = cv2.imread(filename)
    img = cv2.resize(img,(image_size,image_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    batch_prediction = model.predict(img_array)

    pred = np.argmax(batch_prediction[0])

    if pred==0:
        result="Normal"
        prev = prevention["normal"]
    else:
        result = "Cataract"
        prev = prevention["cataract"]
    res = batch_prediction[0]
    res.sort()
    res = res[::-1]
    prob = round(res[0]*100, 2)

    return result, prob, prev
def speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
@app.route('/')
def home():
        return render_template("home.html")
@app.route('/result' , methods = ['GET' , 'POST'])
def result():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                result , prob, prev = predictResult(img_path , model)

                predictions = {
                    "class":result,
                    "prob": prob,
                    "prev": prev,
                }

            else:
                error = "Please upload images of jpg extension only"

            if(len(error) == 0):
                return render_template('result.html' , img  = img , predictions = predictions, func = speech(prev))
            else:
                return render_template('home.html' , error = error)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug = True)


