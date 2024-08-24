# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8

# Install OS packages for Pillow-SIMD
RUN yum -y install tar gzip zlib freetype-devel \
    gcc \
    ghostscript \
    lcms2-devel \
    libffi-devel \
    libimagequant-devel \
    libjpeg-devel \
    libraqm-devel \
    libtiff-devel \
    libwebp-devel \
    make \
    openjpeg2-devel \
    rh-python36 \
    rh-python36-python-virtualenv \
    sudo \
    tcl-devel \
    tk-devel \
    tkinter \
    which \
    xorg-x11-server-Xvfb \
    zlib-devel \
    && yum clean all

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.8 -m pip install -r requirements.txt
# Replace Pillow with Pillow-SIMD to take advantage of AVX2
#RUN pip uninstall -y pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# Copy the earlier created app.py file to the container
COPY app.py ./

# Download ResNet50 and store it in a directory
RUN mkdir model
RUN curl -k -L https://www.kaggle.com/models/google/resnet-v2/TensorFlow2/101-classification/2?tf-hub-format=compressed -o ./model/resnet.tar.gz

# Extract the .tar.gz file into the /model directory
RUN tar -xzvf model/resnet.tar.gz -C ./model

RUN rm -r model/resnet.tar.gz

# Download ImageNet labels
RUN curl -k -L https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt -o ./model/ImageNetLabels.txt

# Set the CMD to your handler
CMD ["app.lambda_handler"]