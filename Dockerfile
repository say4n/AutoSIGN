# Base image heroku cedar stack v14
FROM heroku/miniconda


# Make folder structure
RUN mkdir /app
RUN mkdir /app/.heroku
RUN mkdir /app/.heroku/vendor
ENV LD_LIBRARY_PATH /app/.heroku/vendor/lib/

RUN apt-get update && apt-get install -y apt-utils 
RUN apt-get update && apt-get install -y gcc 
# RUN apt-get update && apt-get install -y g++
RUN apt-get update && apt-get install -y make 
RUN apt-get update && apt-get install -y cmake 
RUN apt-get update && apt-get install -y libopenblas-dev
WORKDIR /app/.heroku


# Install latest setup-tools and pip
RUN curl -s -L https://bootstrap.pypa.io/get-pip.py > get-pip.py
RUN python get-pip.py
RUN rm get-pip.py


# Install other deps
RUN pip install flask

RUN conda install -y opencv scipy=0.18.0 pillow=3.0.0
RUN pip install Theano
RUN pip install https://github.com/Lasagne/Lasagne/archive/master.zip
# RUN pip install opencv-python

RUN pip install gunicorn
RUN pip install eventlet


WORKDIR /app/
ADD . /app/


CMD gunicorn --worker-class eventlet -w 1 main:app