from __future__ import print_function
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

 # Process_Folder
import tensorflow as tf

from scipy.misc import imread
from preprocess.normalize import preprocess_signature

#import signet
import tf_signet
#from cnn_model import CNNModel
from tf_cnn_model import TF_CNNModel

import numpy as np
import sys
import scipy.io



UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

THRESHOLD = 10
DEBUG = True

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit maximum allowed payload to 16 megabytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Thresholds

main_thr_1 = 0.0 # Normal
main_thr_2 = 0.0 # High
main_thr_3 = 0.0 # Very High

same_upper  = 0.0
same_middle = 0.0
same_lower  = 0.0

forg_upper  = 0.0
forg_middle = 0.0
forg_lower  = 0.0

diff_upper  = 0.0
diff_middle = 0.0
diff_lower  = 0.0


# level 0 = Normal, 1 = High, 2 = Very High
def compare_signatures(path1,path2,level):

    canvas_size = (952, 1360)
    max1 = 0
    max2 = 0

    # Load the model
    model_weight_path = 'models/signet.pkl'
    #model = TF_CNNModel(signet, model_weight_path)
    model = TF_CNNModel(tf_signet, model_weight_path)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    original1 = imread(path1, flatten=1)
    processed1 = preprocess_signature(original1, canvas_size)

    original2 = imread(path2, flatten=1)
    processed2 = preprocess_signature(original2, canvas_size)

    feature_vector1 = model.get_feature_vector(sess,processed1)
    feature_vector2 = model.get_feature_vector(sess,processed2)
    feature_vector1 = feature_vector1.T
    feature_vector2 = feature_vector2.T

    dist = (abs(feature_vector1**2 - feature_vector2**2))**(0.5)
    #print(dist)

    for idx, val in enumerate(dist):
        if np.isnan(val):
            dist[idx] = 0

    dist = np.sum(dist)

    main_thr = 0.0
    if level is 0:
        main_thr = main_thr_1
    elif level is 1:
        main_thr = main_thr_2
    elif level is 3:
        main_thr = main_thr_3

    decision = -1

    if(dist<main_thr):
        decision = 1
    else:
        decision = 0

    same_per = 0.0
    forg_per = 0.0
    diff_per = 0.0


    return dist,decision,same_per,forg_per,diff_per


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
        try:
            signatureA = request.files.get("signatureA")
            signatureB = request.files.get("signatureB")

            security_lvl = request.form.get("security")

            filenameA = secure_filename(signatureA.filename)
            signature_pathA = os.path.join(app.config['UPLOAD_FOLDER'], filenameA)
            signatureA.save(signature_pathA)

            filenameB = secure_filename(signatureB.filename)
            signature_pathB = os.path.join(app.config['UPLOAD_FOLDER'], filenameB)
            signatureB.save(signature_pathB)

            security_lvl = int(security_lvl)

            dist, decision, same_percent, forg_percent, diff_percent = compare_signatures(signature_pathA,
                                                                                          signature_pathB,
                                                                                          security_lvl)

        except:
            flash(u'An error occured, please try again!', 'error')
            return redirect("/")

        if DEBUG:
            print("type(signatureA): ", type(signatureA))
            print("type(signatureB): ", type(signatureB))
            print("type(security_lvl): ", type(security_lvl))

        return render_template("result.html", dist=dist)



if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    app.run(debug=DEBUG, host='0.0.0.0')

