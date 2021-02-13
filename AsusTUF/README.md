# System Info
2021 Asus TUF Gaming A15, Windows 10 and Ubuntu 20.04.2 dual systems

CPU: AMD Ryzen7 5800H

GPU: GeForce RTX 3070 Laptop

Memory: 16GB

# NVIDIA

* Driver (460.39): 
  Three different ways to install the NVIDIA-driver in ubuntu, refer to [Install NVIDIA drive on Ubuntu 20.04](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux).
  *GUI or Command-line intallation is preferred.*

  To solve the compatibility problem between AMD Ryzen7 5xxx series CPU and RTX 3xxx series GPU, use the [procedure](https://forums.developer.nvidia.com/t/ubuntu-mate-20-04-with-rtx-3070-on-ryzen-5900-black-screen-after-boot/167681/30).

* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (11.0): 
  Note that Cuda toolkit need to be compatible with TensorFlow. Check the [table](https://www.tensorflow.org/install/source#gpu)

  Since the NVIDIA driver is already installed, recommend to install Cuda Toolkit using runfile, and make sure to un-select install driver during the installation.
  
  Because the latest tensorflow 2.4.0 supports CUDA 11.0, cudnn 8.0, and python 3.6-3.8, CUDA 11.0 is installed using runfile. 
  
  Installation Instructions:

      wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
      sudo sh cuda_11.0.2_450.51.05_linux.run
  
  The output is:
  
      ===========
      = Summary =
      ===========

      Driver:   Not Selected
      Toolkit:  Installed in /usr/local/cuda-11.0/
      Samples:  Installed in /home/yohann/, but missing recommended libraries

      Please make sure that
      -   PATH includes /usr/local/cuda-11.0/bin
      -   LD_LIBRARY_PATH includes /usr/local/cuda-11.0/lib64, or, add /usr/local/cuda-11.0/lib64 to /etc/ld.so.conf and run ldconfig as root

      To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.0/bin

      Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-11.0/doc/pdf for detailed information on setting up CUDA.
      ***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least .00 is required for CUDA 11.0 functionality to work.
      To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
          sudo <CudaInstaller>.run --silent --driver

      Logfile is /var/log/cuda-installer.log
   
  Update PATH and LD_LIBRARY_PATH:
  
      export PATH=/usr/local/cuda-11.0/bin:$PATH
      export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
   
* [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (8.0.5): 
  Based on system and cuda version, choose cuDNN v8.0.5 for CUDA 11.0 through Debian Installation.

  Instruction: 
  Before issuing the following commands, you'll need to replace x.x and 8.x.x.x with your specific CUDAand cuDNN versions and package date.

      Navigate to your <cudnnpath> directory containing the cuDNN Debian file.
      Install the runtime library, for example:
        sudo dpkg -i libcudnn8_x.x.x-1+cudax.x_amd64.deb
      or
        sudo dpkg -i libcudnn8_x.x.x-1+cudax.x_arm64.deb
      Install the developer library, for example:
        sudo dpkg -i libcudnn8-dev_8.x.x.x-1+cudax.x_amd64.deb
      or
        sudo dpkg -i libcudnn8-dev_8.x.x.x-1+cudax.x_arm64.deb
      Install the code samples and the cuDNN library documentation, for example:
        sudo dpkg -i libcudnn8-samples_8.x.x.x-1+cudax.x_amd64.deb
      or
        sudo dpkg -i libcudnn8-samples_8.x.x.x-1+cudax.x_arm64.deb

# Anaconda
  Install Anaconda, follow the official instruction [here](https://docs.anaconda.com/anaconda/install/linux/)

# OpenCV
  [OpenCV](https://opencv.org/releases/)
  [OpenCV-contrib](https://github.com/opencv/opencv_contrib/releases) 
  
  Prerequisite

    [compiler] sudo apt-get install build-essential
    [required] sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
    [optional] sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

  Install OpenCV
  
  	cmake -D CMAKE_BUILD_TYPE=RELEASE -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.12/modules -D WITH_LIBV4L=ON -D WITH_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=ON -D WITH_NVCUVID=ON -D CUDA_GENERATION=Auto -D WITH_TBB=ON -D WITH_OPENMP=ON -D WITH_OPENGL=ON -D ENABLE_CXX11=ON -D OPENCV_ENABLE_NONFREE=ON -D ENABLE_PRECOMPILED_HEADERS=OFF ..
    
  Problems Encountered:
  
  1. E: Unable to locate package libjasper-dev
  
    sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
    sudo apt update
    sudo apt install libjasper-dev
  
  2. python error: no module named 'cv2'
  
    pip install opencv-python
  
  

