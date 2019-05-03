# Define Base Image
# FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Install dependencies

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

# Install ceres solver
RUN cd /opt
WORKDIR /opt
RUN apt-get install -y libatlas-base-dev libsuitesparse-dev libgoogle-glog-dev libeigen3-dev libsuitesparse-dev
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN cd ceres-solver
WORKDIR /opt/ceres-solver
RUN mkdir build
RUN cd build
WORKDIR /opt/ceres-solver/build
RUN cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
RUN make
RUN make install


# Install Colmap

RUN cd /opt
WORKDIR /opt
RUN git clone https://github.com/colmap/colmap
RUN cd colmap
WORKDIR /opt/colmap
RUN mkdir build
RUN cd build
WORKDIR /opt/colmap/build
RUN cmake ..
RUN make
RUN make install
RUN cd /

# RUN mkdir -p /home/app
# RUN mkdir /home/app/data

# # Specify working directory

# WORKDIR /home/app

# # Copy script from Host machine to working directory of the container
# COPY colmap.sh .

# # Specify entry point of the container

# ENTRYPOINT ["sh", "colmap.sh"]




# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        ffmpeg \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install --upgrade ipython && \
    pip --no-cache-dir install \
        ipykernel \
        jupyter \
        jupyterlab \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        scikit-image \
        imageio==2.4.0 \
        && \
    python -m ipykernel.kernelspec


# Install TensorFlow CPU version from central repo
RUN pip --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.13.1-cp27-none-linux_x86_64.whl
# https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Install OpenCV
RUN apt-get update && apt-get install -y libopencv-dev python-opencv && \
    echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc
    
# Install GLFW
RUN apt-get update && apt-get install -y libglfw3-dev

RUN apt-get install -y imagemagick

# Expose Ports for Tensorboard (6006)
EXPOSE 6006

# #Jupyter notebook related configs
# COPY jupyter_notebook_config.py /root/.jupyter/
EXPOSE 8888

# # Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
# COPY run_jupyter.sh /home/

# #Adding flask
# RUN pip install flask
# EXPOSE 5000

RUN mkdir -p /workspace
WORKDIR /workspace
RUN chmod -R a+w /workspace