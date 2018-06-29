from __future__ import print_function
import os
from flask import Flask, flash, request, redirect, url_for, render_template, render_template_string, send_from_directory
from datetime import datetime
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_user import login_required, UserManager, UserMixin, SQLAlchemyAdapter, current_user
from flask_migrate import Migrate

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
from PIL import Image



UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

THRESHOLD = 10
DEBUG = True

app = Flask(__name__)

app.secret_key = "ultra super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit maximum allowed payload to 16 megabytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.config['CSRF_ENABLED'] = True
app.config['USER_ENABLE_EMAIL'] = False

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./autosign.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False



db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False, server_default='')
    active = db.Column(db.Boolean(), nullable=False, server_default='0')
    tests = db.relationship('Test', backref='Test', lazy='dynamic')


db_adapter = SQLAlchemyAdapter(db, User)
user_manager = UserManager(db_adapter, app)

'''
class Sign_image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(50), nullable=False, unique=True)
    tests = db.relationship('Test', backref='Test', lazy='dynamic')

    def __repr__(self):
        return '<Post {}>'.format(self.id)



class Test(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    res_dist = db.Column(db.Float)
    res_decsn = db.Column(db.Integer)
    res_same_per = db.Column(db.Float)
    res_forg_per = db.Column(db.Float)
    res_diff_per = db.Column(db.Float)

    signature_1 = db.Column(db.Integer, db.ForeignKey('sign_image.id'))
    signature_2 = db.Column(db.Integer, db.ForeignKey('sign_image.id'))

    def __repr__(self):
        return '<Post {}>'.format(self.id)
'''


class Test(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    res_dist = db.Column(db.Float)
    res_decsn = db.Column(db.Boolean)
    res_same_per = db.Column(db.Float)
    res_forg_per = db.Column(db.Float)
    res_diff_per = db.Column(db.Float)
    signature_1 = db.Column(db.String(50), nullable=False)
    signature_2 = db.Column(db.String(50), nullable=False)
    errors = db.relationship('Error', backref='Error', lazy='dynamic')


    def __repr__(self):
        return '<Test {}>'.format(self.id)

class Error(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    ref_test = db.Column(db.Integer, db.ForeignKey('test.id'))
    comment = db.Column(db.String(500), nullable=False)
    flag = db.Column(db.Integer)

# Thresholds

main_thr_1 = 920.0 # Normal
main_thr_2 = 880 # High
main_thr_3 = 800.0 # Very High

same_upper  = 1050.0
same_middle = 850.0
same_lower  = 650.0

forg_upper  = 1250.0
forg_middle = 1120.0
forg_lower  = 810.0

diff_upper  = 1600.0
diff_middle = 1200.0
diff_lower  = 1000.0


# level 0 = Normal, 1 = High, 2 = Very High
def compare_signatures(path1,path2,level):

    canvas_size = (952, 1360)
    max1 = 0
    max2 = 0

    tf.reset_default_graph()

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
    decision = -1

    if level is 0:
        main_thr = main_thr_1
    elif level is 1:
        main_thr = main_thr_2
    elif level is 2:
        main_thr = main_thr_3

    if(dist<main_thr):
        decision = 1
    else:
        decision = 0

    same_per = 0.0
    forg_per = 0.0
    diff_per = 0.0

    # Calculating same_per
    if(dist<same_lower):
        same_per = 100 - ((dist-0)/(same_lower-0))*5.0
    elif(dist<same_middle):
        same_per = 95 - ((dist-same_lower)/(same_middle-same_lower))*45
    elif(dist<same_upper):
        same_per = 50 - ((dist-same_middle)/(same_upper-same_middle))*45
    elif(dist>1350):
        same_per = 0
    elif(dist>same_upper):
        same_per = 5 - ((dist-same_upper)/(1350-same_upper))*5

    # Calculating forg_per
    if((dist<forg_lower)&(dist>=700)):
        forg_per = ((dist-700)/(forg_lower-700))*15
    elif(dist<700):
        forg_per = 0.0
    elif(dist<forg_middle):
        forg_per = 15 + ((dist-forg_lower)/(forg_middle-forg_lower))*60
    elif(dist<forg_upper):
        forg_per = 15 + ((dist-forg_middle)/(forg_upper-forg_middle))*60
    elif(dist>=2000):
        forg_per = 0.0
    elif(dist>forg_upper):
        forg_per = ((dist-forg_upper)/(2000-forg_upper))*15

    # Calculating diff_per
    if(dist<=1000):
        diff_per = 0.0
    elif(dist<diff_lower):
        diff_per = ((dist-1000)/(diff_lower-1000))*5.0
    elif(dist<diff_middle):
        diff_per = 5 + ((dist-diff_lower)/(diff_middle-diff_lower))*45
    elif(dist<diff_upper):
        diff_per = 50 + ((dist-diff_middle)/(diff_upper-diff_middle))*45
    elif(dist>diff_upper):
        diff_per = 95 + ((dist-same_upper)/(3000-same_upper))*5

    if(dist>=3000):
        same_per = 0.0
        forg_per = 0.0
        diff_per = 100.0

    same_per = float("{0:.2f}".format(same_per))
    forg_per = float("{0:.2f}".format(forg_per))
    diff_per = float("{0:.2f}".format(diff_per))

    return dist, decision, same_per, forg_per, diff_per


@app.route("/")
@login_required
def index():
    return render_template("index.html", username=current_user.username)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/dashboard/")
@login_required
def dashboard():
    return render_template("dashboard.html",
                           username=current_user.username,
                           all_tests=current_user.tests.order_by(Test.timestamp.desc()).all())

@app.route("/error_handling/")
@login_required
def errors():
    return render_template("flags.html",
                           username=current_user.username,
                           flags=Error.query.all(),
                           tests=Test.query.all())


@app.route("/flag_report", methods=["POST"])
def flag_endpoint():
    try:
        test_id = request.form.get("id")
        comment = request.form.get("comment")
        flag = request.form.get("flag")

        err = Error(ref_test=test_id, comment=comment, flag=flag)

        db.session.add(err)
        db.session.commit()
    except Exception as e:
        print(e)
        abort(400)


@app.route('/image/<path:filename>')
def image(filename):
    try:
        w = int(request.args['w'])
        h = int(request.args['h'])
    except (KeyError, ValueError):
        return send_from_directory(UPLOAD_FOLDER, filename)

    try:
        im = Image.open(os.path.join(UPLOAD_FOLDER, filename))
        im.thumbnail((w, h), Image.ANTIALIAS)
        io = StringIO.StringIO()
        im.save(io, format='JPEG')
        return Response(io.getvalue(), mimetype='image/jpeg')

    except IOError:
        abort(404)

    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/verify/", methods=["POST"])
@login_required
def verify():
    """
    accepts POST of json data

    data = {
        "signature_image" : image
        "uuid" : uuid
    }
    """
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

        test_ = Test(user_id=current_user.id,
                     res_dist=dist,
                     res_decsn=bool(decision),
                     res_same_per=same_percent,
                     res_forg_per=forg_percent,
                     res_diff_per=diff_percent,
                     signature_1=filenameA,
                     signature_2=filenameB)

        db.session.add(test_)
        db.session.commit()

    except Exception as e:
        print(e)
        flash(u'An error occured, please try again!', 'error')
        return redirect("/")

    if DEBUG:
        print("type(signatureA): ", type(signatureA))
        print("type(signatureB): ", type(signatureB))
        print("type(security_lvl): ", type(security_lvl))

    return render_template("result.html",
                           dist=dist,
                           decision=bool(decision),
                           same_percent=same_percent,
                           forg_percent=forg_percent,
                           diff_percent=diff_percent,
                           username=current_user.username)


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    app.run(debug=DEBUG, host='0.0.0.0')

