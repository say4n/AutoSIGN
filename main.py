from __future__ import print_function
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

# # # Process_Folder
# import tensorflow as tf

# from scipy.misc import imread
# from preprocess.normalize import preprocess_signature

# #import signet
# import tf_signet
# #from cnn_model import CNNModel
# from tf_cnn_model import TF_CNNModel

# import numpy as np
# import sys
# import scipy.io


UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

THRESHOLD = 10

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit maximum allowed payload to 16 megabytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# def compare_signatures(path1,path2):

#     canvas_size = (952, 1360)
#     max1 = 0
#     max2 = 0

#     # Load the model
#     model_weight_path = 'models/signet.pkl'
#     #model = TF_CNNModel(signet, model_weight_path)
#     model = TF_CNNModel(tf_signet, model_weight_path)

#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())

#     original1 = imread(path1, flatten=1)
#     processed1 = preprocess_signature(original1, canvas_size)

#     original2 = imread(path2, flatten=1)
#     processed2 = preprocess_signature(original2, canvas_size)

#     feature_vector1 = model.get_feature_vector(sess,processed1)
#     feature_vector2 = model.get_feature_vector(sess,processed2)
#     feature_vector1 = feature_vector1.T
#     feature_vector2 = feature_vector2.T

#     dist = (abs(feature_vector1**2 - feature_vector2**2))**(0.5)
#     #print(dist)

#     for idx, val in enumerate(dist):
#         # print(val.shape)
#         if np.isnan(val):
#             dist[idx] = 0

#     dist = np.sum(dist)
#     #print(dist)

#     return dist


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

        try:
            signatureA = request.form.get("signatureA")
            signatureB = request.form.get("signatureB")
            security_lvl = request.form.get("security")
        except:
            flash(u'An error occured, please try again!', 'error')
            return redirect("/")

        print("type(signatureA): ", type(signatureA))
        print("type(signatureB): ", type(signatureB))

        return render_template("result.html")



if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    app.run(debug=True, host='0.0.0.0')

