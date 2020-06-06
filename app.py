from flask import Flask, json, request, url_for
from flask_cors import CORS, cross_origin
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/src/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
target_size = (96, 96)
model_path = os.path.dirname(os.path.abspath(__file__)) + "/src/my_model1.keras"
api = Flask(__name__)
cors = CORS(api)
api.config['CORS_HEADERS'] = 'Content-Type'
api.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS

def predict(filename):
    if filename:
        img = Image.open(UPLOAD_FOLDER + '/' + filename).convert('RGB').resize(target_size)
        print('[INFO] image loaded successfully')
        print("[INFO] loading the model ...")
        model = load_model(model_path)
        print("[INFO] model loaded successfully")
        print("[INFO] Starting the prediction...")
        img = np.array(img) / 255.0
        img = img.reshape((1, img.shape[0], img.shape[1],
                               img.shape[2]))
        preds = model.predict(img)
        print("[INFO] prediction success ", preds)
        result = preds.argmax(axis=-1)[0]
        print("[INFO] final result ", result)
        return str(result)
        # return str(3)
    return 'error'


@api.route('/upload', methods=['POST'])
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
            file.save(os.path.join(api.config['UPLOAD_FOLDER'], filename))
            print("[INFO] Save it to:", UPLOAD_FOLDER)
            print("[INFO] Starting the prediction with the Keras model...")
            res_prediction = predict(filename)
            print('[INFO] the keras model returned ', res_prediction)
            if res_prediction:
                return res_prediction
    return 'error'

if __name__ == '__main__':
    api.run()