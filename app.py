from flask import Flask, json, request, url_for
from flask_cors import CORS, cross_origin
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/src/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
target_size = 96
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
        image = cv2.imread(UPLOAD_FOLDER + '/' + filename)
        print('[INFO] image loaded successfully')
        image = cv2.resize(image, (target_size, target_size))
        print('[INFO] image resized successfully')
        image = image.astype("float") / 255.0
        image = image.reshape((1, image.shape[0], image.shape[1],
                               image.shape[2]))
        print("[INFO] loading the model ...")
        model = load_model(model_path)
        preds = model.predict(image)
        result = preds.argmax(axis=1)[0]
        # return str(result)
        return str(3)
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