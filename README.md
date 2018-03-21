# AutoSIGN
A Signature Validation and Mandate Verification System by using Siamese Networks and One-Shot Learning. 

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

They can be installed by running the following commands: 

```
conda install -y opencv "scipy=0.18.0" "pillow=3.0.0"
conda install -y jupyter notebook matplotlib # Optional, to run the example in jupyter notebook
pip install "Theano==0.9"
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
pip install opencv-python
pip install flask
```

This code was tested in Ubuntu 16.04 and <Mac-version>.
Please open an Issue if you get into any problems. 

## Downloading the models

Simply run the following code: 
```
git clone https://github.com/Not-Boring/AutoSIGN.git
cd AutoSIGN/models
wget "https://storage.googleapis.com/luizgh-datasets/models/signet_models.zip"
unzip signet_models.zip
``` 

## Develop

The project uses [`pipenv`](https://docs.pipenv.org/) for dependency management, install it.
Then launch a shell with `pipenv shell`. 

To run locally, type in `python -B main.py`.

## Test

Run `python_example.py`. This script pre-process a signature, and compares the feature vector obtained by the model to the results obtained previously. 

Visit https://auto-sign.herokuapp.com !

## Deploy

Deploy using the provided Dockerfile!

## Team

- Sayan Goswami
- Ayan Sinha Mahaptra
- Dibyadip Chatterjee
- Arpan Bhowmik