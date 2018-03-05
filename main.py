from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World!"

@app.route("/verify", methods=["POST"])
def verify():
    """
    accepts POST of json data

    data = {
        "link_to_uploaded_signature" : url
        "uid_of_customer" : uid
    }
    """

    if request.method == "POST":
        url = request.form.get("link_to_uploaded_signature")
        uid = request.form.get("uid_of_customer")
        
        if urlparse(url).scheme in ["http", "https"]:
            
        else:
            abort(400)


if __name__ == "__main__":
    app.run(debug=True)
