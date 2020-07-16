from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
from os import path
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/src/uploads"
DownloadFolder = os.path.dirname(os.path.abspath(__file__)) + "/src/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
target_size = (96, 96)
model_path = os.path.dirname(os.path.abspath(__file__)) + "/src/modelTL2.keras"
print("Final model path = " + model_path)
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS



def imageToArray(image):
    image = image.convert('RGB')
    image = image.resize((200,200))
    e = np.array(image) / 255
    e = e.reshape(1, e.shape[0], e.shape[1], 3)
    return e


def predict(filename):
    if filename:
        image = Image.open(UPLOAD_FOLDER + '/' + filename)
        print('[INFO] image loaded successfully')
        print("[INFO] model loaded successfully")

        imageArray = imageToArray(image)
        print("[INFO] convertion to array successfully...")

        model = load_model(model_path)

        preds = model.predict(imageArray)
        # print("preds =" + preds.argmax(axis=-1))
        result = preds.argmax(axis=-1)[0]
        print("[INFO] final result ", result)
        return str(result)
    return 'error'



@app.route("/")
@cross_origin()
def index():
    return "<h1>Weclome to esgi-prediction, use this web-server to predict Dog, Cat and Parrot using the endpoint /upload</h1>"

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    for upload in request.files.lists():
        file = request.files[upload[0]]
        if file.filename == '':
            print('[INFO] fileName empty')
            return 'failure'
        if file and allowed_file(file.filename):
            print("[INFO] File supported moving on...")
            print("[INFO] Accept incoming file:", file.filename)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("[INFO] Save it to:", UPLOAD_FOLDER)
            print("[INFO] Starting the prediction with the Keras model...")
            res_prediction = predict(filename)
            print('[INFO] the keras model returned ', res_prediction)
            if res_prediction:
                return res_prediction
    return 'error'

def downloadSource():
    if not path.exists(model_path):
        print("Download ./prediction/ressource/ File From google drive : ")
        gdd.download_file_from_google_drive(file_id = '1O3GMPr9kk1hyV2jFNGdtsc9Ne5G_lwK6',
                                            dest_path = DownloadFolder + 'ressource.zip', unzip=True)
        os.remove(DownloadFolder + "/ressource.zip")


if __name__ == '__main__':
    downloadSource()
    app.run()
