import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

# # Process_Folder

# from scipy.misc import imread
# from preprocess.normalize import preprocess_signature
# import signet
# from cnn_model import CNNModel
# import numpy as np
# import sys
# import scipy.io


UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

THRESHOLD = 10

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit maximum allowed payload to 16 megabytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def compare_signatures(path1,path2):

    canvas_size = (952, 1360)
    max1 = 0
    max2 = 0

    # Load the model
    model_weight_path = 'models/signet.pkl'
    model = CNNModel(signet, model_weight_path)

    original1 = imread(path1, flatten=1)
    processed1 = preprocess_signature(original1, canvas_size)

    original2 = imread(path2, flatten=1)
    processed2 = preprocess_signature(original2, canvas_size)

    feature_vector1 = model.get_feature_vector(processed1)
    feature_vector2 = model.get_feature_vector(processed2) 
    feature_vector1 = feature_vector1.T
    feature_vector2 = feature_vector2.T 

    dist = (abs(feature_vector1**2 - feature_vector2**2))**(0.5)
    #print(dist)

    for idx, val in enumerate(dist):
        # print(val.shape)
        if np.isnan(val):
            dist[idx] = 0

    dist = np.sum(dist)
    #print(dist)

    return dist


@app.route("/")
def index():
    return render_template("index.html")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/verify", methods=["POST"])
def verify():
    """
    accepts POST of json data

    data = {
        "signature_image" : image
        "uuid" : uuid
    }
    """
    if request.method == "POST":
        # check if the post request has the file part
        if 'signature_image' not in request.files:
            flash(u'No file was uploaded, please try again!', 'error')
            return redirect("/")

        uuid = request.form.get('uuid')

        file = request.files['signature_image']
        gt_file = request.files['signature_gt_image']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '' or gt_file.filename == '':
            flash(u'Empty image uploaded, please try again!', 'error')
            return redirect("/")
        
        if file and allowed_file(file.filename) and gt_file and allowed_file(gt_file.filename):
            filename = secure_filename(file.filename)
            fp1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fp1)

            gt_filename = secure_filename(gt_file.filename)
            fp2 = os.path.join(app.config['UPLOAD_FOLDER'], gt_filename)
            gt_file.save(fp2)

            # result = compare_signatures(fp1, fp2)

            return render_template("result.html", dist=100)


if __name__ == "__main__":
    app.run(debug=False)

