import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit maximum allowed payload to 16 megabytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            gt_filename = secure_filename(gt_file.filename)
            gt_file.save(os.path.join(app.config['UPLOAD_FOLDER'], gt_filename))

            return filename


if __name__ == "__main__":
    app.run(debug=True)
