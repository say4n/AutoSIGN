# AutoSIGN - Proof of Concept
A Signature Validation and Mandate Verification System by using Siamese Networks and One-Shot Learning. 
Head out to http://130.162.78.201 for our demo. :) 

# Installation

## Pre-requisites 

The code is written in Python 2. We recommend using the Anaconda python distribution, and create a new environment using: 
```
conda create -n AutoSIGN -y python=2
source activate AutoSIGN
```

The following libraries are required

* Scipy version 0.18
* Pillow version 3.0.0
* OpenCV
* Theano
* Lasagne
* Tensorflow
* Flask
* libgtk2.0-dev
* libsm6 
* libxext6
* flask_user
* flask_sqlalchemy
* datetime

They can be installed by running the following commands: 

```
conda install -y opencv "scipy=0.18.0" "pillow=3.0.0"
conda install -y jupyter notebook matplotlib # Optional, to run the example in jupyter notebook
pip install "Theano==0.9"
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
pip install opencv-python
pip install flask
pip install flask_migrate
pip install datetime
sudo apt-get update && sudo apt-get install libgtk2.0-dev
sudo apt update && sudo apt install -y libsm6 libxext6
```

This code was tested in Ubuntu 16.04 and <Mac-version>.
Please open an Issue if you get into any problems. 

## Downloading the models

Simply run the following code: 
```
git clone https://github.com/Not-Boring/AutoSIGN.git
cd AutoSIGN/models
wget "https://storage.googleapis.com/luizgh-datasepython ts/models/signet_models.zip"
unzip signet_models.zip
``` 

## Source File(s)

### main.py

  #### Dependencies
   * numpy
   * scipy
   * tensorflow
   * flask
   * flask_user
   * flask_sqlalchemy
   * PIL
   * flask_migrate
   * datetime

  #### Functions
   * compare_signatures : Accepts paths to two images, computes the similarity between them

  #### API Routes
   * /: Landing page view. This is also the Test page.
   * /verify: Signature verification endpoint accepts POST requests with payloads `signature_image` and
              `signature_gt_image`, they being either jpeg or png images. Then it passes those images through the system and
              the results are stored in a new Test User. User statistics are also updated here.
   * /dashboard: All the tests performed by the user is shown here. This also has error reporting system for the tests.
   * /reports:  All the Error reports are listed in this page with every information of the test they refer to.
   * /flag_report: This saves an error report to the Database in a Error class.
  
  #### Database Models
   * User: Model to Store User Information
   * Test: Model to Store Test Reports
   * Error: Model to store Error Reports
   

### tf_CNN.py

  #### Dependencies
   * tensorflow
   * numpy
   * cPickle
  #### Functions
   * `_init_` : Initializes the CNN.
   * `get_feature_vector` : Sends an image through one forward pass of the CNN and outputs the Feature Vector required to 
                          compare signatures.
          
          
### tf_signet.py

  #### Dependencies
   * tensorflow 
   * Tensorflow-slim
  #### Functions
   * `build_architecture` : Builds the CNN architecture to be used and loads the baseline pre-trained weights using the 
                          following 3 base functions. 
   * `conv_bn` : Implements Convolutional Layers.
   * `dense_bn` : Implements Fully-Connected Layers.
   * `batch_norm` : Implements Batch Normalization to be used in the above functions.

### tf_example.py
This File compares some results with the ones obtained by the us,to ensure consistency in all the dependencies used throughout the project. 

### normalize.py

  #### Dependencies
   * cv2
   * numpy
   * scipy
   * tesseract 

  #### Functions
   * `preprocess_signature`: Uses the following three functions to pre-process the signature images into the fixed input 
                             size of the CNN. This does all the centering, noise-removal and everything else that is
                             necessary for the Model to successfully learn from/process the signatures.
                           
   * `Crop_center`,`resize_image` and `normalize_image`: 

### lasagne_to_tf.py

  #### Dependencies
   * numpy

  #### Functions
There are three classes and corresponding initialization functions to change the model from Lasagne to Tensorflow, as 
Lasagne uses the format BCHW, while tensorflow uses BHWC, and also there is some difference between the Convolution
Filters as they are flipped with respect to each other.

These functions implements those changes to load the pre-trained model we use as our baseline. 


### templates/
   #### Html Files
   base.html
   head.html
   drawer.html
   nav_header.html
   index.html
   dsahboard.html
   result.html
   flags.html
   
### Database : autosign.db 
 A sqlite3 database to store data according to our models.
  

## Develop

The project uses [`pipenv`](https://docs.pipenv.org/) for dependency management, install it.
Then launch a shell with `pipenv shell`. 

To run locally, type in `python -B main.py`.

## Test

Run `tf_example.py`. This script pre-process a signature, and compares the feature vector obtained by the model to the results obtained previously. 

Visit https://auto-sign.herokuapp.com !

## Deploy

Deploy using the provided Dockerfile!

## Team

- Sayan Goswami
- Ayan Sinha Mahaptra
- Dibyadip Chatterjee
- Arpan Bhowmik
