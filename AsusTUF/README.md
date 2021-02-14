# Table of Content
- [System Info](#system-info)
- [NVIDIA](#nvidia)
- [OpenCV](#opencv)
  * [Problems](#problems-encountered)
- [Anaconda](#anaconda)
- [NOCS](#nocs-network)
  * [Problems](#problems-encountered)
- [Libfusion](#libfusion)


# System Info
2021 Asus TUF Gaming A15, Windows 10 and Ubuntu 20.04.2 dual systems

CPU: AMD Ryzen7 5800H

GPU: GeForce RTX 3070 Laptop

Memory: 16GB

[Back to Top](#table-of-content)


# NVIDIA
* Driver (460.39): 
  Three different ways to install the NVIDIA-driver in ubuntu, refer to [Install NVIDIA drive on Ubuntu 20.04](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux).
  *GUI or Command-line intallation is preferred.*

  To solve the compatibility problem between AMD Ryzen7 5xxx series CPU and RTX 3xxx series GPU, use the [procedure](https://forums.developer.nvidia.com/t/ubuntu-mate-20-04-with-rtx-3070-on-ryzen-5900-black-screen-after-boot/167681/30).
0
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
      
  To delete current CUDA:
      
      cd /usr/local/cuda-11.0/bin
      chmod +x cuda-uninstaller
      sudo ./cuda-uninstaller

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

[Back to Top](#table-of-content)


# OpenCV
  Links to download OpenCV source codes:
  [OpenCV](https://opencv.org/releases/),
  [OpenCV-contrib](https://github.com/opencv/opencv_contrib/releases).
  
  Prerequisite

    [compiler] sudo apt-get install build-essential
    [required] sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
    [optional] sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

  Install OpenCV
  
  	cmake -D CMAKE_BUILD_TYPE=RELEASE -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.13/modules -D WITH_CUDA=ON -D CUDA_ARCH_BIN=8.0 -D WITH_CUDNN=OFF -D BUILD_opencv_cudacodec=OFF -D OPENCV_ENABLE_NONFREE=ON -DBUILD_JAVA=OFF -DBUILD_opencv_java_bindings_generator=OFF -D BUILD_PYTHON=OFF -D BUILD_opencv_python_bindings_generator=OFF -D WITH_QT=ON -D ENABLE_CXX11=ON -D BUILD_TIFF=ON ..
    
  CMake Output:
  
    -- Detected processor: x86_64
    -- Looking for ccache - not found
    -- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found suitable version "1.2.11", minimum required is "1.2.3") 
    Cleaning INTERNAL cached variable: WEBP_LIBRARY
    Cleaning INTERNAL cached variable: WEBP_INCLUDE_DIR
    -- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.2.11") 
    CMake Warning (dev) at /usr/share/cmake-3.16/Modules/FindOpenGL.cmake:275 (message):
      Policy CMP0072 is not set: FindOpenGL prefers GLVND by default when
      available.  Run "cmake --help-policy CMP0072" for policy details.  Use the
      cmake_policy command to set the policy and suppress this warning.

      FindOpenGL found both a legacy GL library:

        OPENGL_gl_LIBRARY: /usr/lib/x86_64-linux-gnu/libGL.so

      and GLVND libraries for OpenGL and GLX:

        OPENGL_opengl_LIBRARY: /usr/lib/x86_64-linux-gnu/libOpenGL.so
        OPENGL_glx_LIBRARY: /usr/lib/x86_64-linux-gnu/libGLX.so

      OpenGL_GL_PREFERENCE has not been set to "GLVND" or "LEGACY", so for
      compatibility with CMake 3.10 and below the legacy GL library will be used.
    Call Stack (most recent call first):
      cmake/OpenCVFindLibsGUI.cmake:76 (find_package)
      CMakeLists.txt:698 (include)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Checking for modules 'gstreamer-base-1.0;gstreamer-video-1.0;gstreamer-app-1.0;gstreamer-riff-1.0;gstreamer-pbutils-1.0'
    --   No package 'gstreamer-base-1.0' found
    --   No package 'gstreamer-video-1.0' found
    --   No package 'gstreamer-app-1.0' found
    --   No package 'gstreamer-riff-1.0' found
    --   No package 'gstreamer-pbutils-1.0' found
    -- Checking for modules 'gstreamer-base-0.10;gstreamer-video-0.10;gstreamer-app-0.10;gstreamer-riff-0.10;gstreamer-pbutils-0.10'
    --   No package 'gstreamer-base-0.10' found
    --   No package 'gstreamer-video-0.10' found
    --   No package 'gstreamer-app-0.10' found
    --   No package 'gstreamer-riff-0.10' found
    --   No package 'gstreamer-pbutils-0.10' found
    -- Checking for modules 'libv4l1;libv4l2'
    --   No package 'libv4l1' found
    --   No package 'libv4l2' found
    -- Looking for linux/videodev.h
    -- Looking for linux/videodev.h - not found
    -- Looking for linux/videodev2.h
    -- Looking for linux/videodev2.h - found
    -- Looking for sys/videoio.h
    -- Looking for sys/videoio.h - not found
    -- Checking for module 'libavresample'
    --   No package 'libavresample' found
    -- Found TBB (cmake): /usr/lib/x86_64-linux-gnu/libtbb.so.2
    -- IPPICV: Download: ippicv_2020_lnx_intel64_20191018_general.tgz
    -- found Intel IPP (ICV version): 2020.0.0 [2020.0.0 Gold]
    -- at: /home/yohann/SLAMs/depend/opencv-3.4.13/build/3rdparty/ippicv/ippicv_lnx/icv
    -- found Intel IPP Integration Wrappers sources: 2020.0.0
    -- at: /home/yohann/SLAMs/depend/opencv-3.4.13/build/3rdparty/ippicv/ippicv_lnx/iw
    -- CUDA detected: 11.0
    -- CUDA: Using CUDA_ARCH_BIN=8.0
    -- CUDA NVCC target flags: -gencode;arch=compute_80,code=sm_80;-D_FORCE_INLINES
    -- Could not find OpenBLAS include. Turning OpenBLAS_FOUND off
    -- Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off
    -- Could NOT find Atlas (missing: Atlas_CLAPACK_INCLUDE_DIR Atlas_CBLAS_LIBRARY Atlas_BLAS_LIBRARY) 
    -- A library with LAPACK API found.
    -- VTK is not found. Please set -DVTK_DIR in CMake to VTK build directory, or to VTK install subdirectory with VTKConfig.cmake file
    -- OpenCV Python: during development append to PYTHONPATH: /home/yohann/SLAMs/depend/opencv-3.4.13/build/python_loader
    -- Caffe:   NO
    -- Protobuf:   NO
    -- Glog:   NO
    -- freetype2:   YES (ver 23.1.17)
    -- harfbuzz:    YES (ver 2.6.4)
    -- HDF5: Using hdf5 compiler wrapper to determine C configuration
    -- Module opencv_ovis disabled because OGRE3D was not found
    -- No preference for use of exported gflags CMake configuration set, and no hints for include/library directories provided. Defaulting to preferring an installed/exported gflags CMake configuration if available.
    -- Failed to find installed gflags CMake configuration, searching for gflags build directories exported with CMake.
    -- Failed to find gflags - Failed to find an installed/exported CMake configuration for gflags, will perform search for installed gflags components.
    -- Failed to find gflags - Could not find gflags include directory, set GFLAGS_INCLUDE_DIR to directory containing gflags/gflags.h
    -- Failed to find glog - Could not find glog include directory, set GLOG_INCLUDE_DIR to directory containing glog/logging.h
    -- Module opencv_sfm disabled because the following dependencies are not found: Eigen Glog/Gflags
    -- Checking for module 'tesseract'
    --   No package 'tesseract' found
    -- Tesseract:   NO
    -- Allocator metrics storage type: 'long long'
    -- HDF5: Using hdf5 compiler wrapper to determine C configuration
    -- data: Download: face_landmark_model.dat
    -- 
    -- General configuration for OpenCV 3.4.13 =====================================
    --   Version control:               unknown
    -- 
    --   Extra modules:
    --     Location (extra):            /home/yohann/SLAMs/depend/opencv_contrib-3.4.13/modules
    --     Version control (extra):     unknown
    -- 
    --   Platform:
    --     Timestamp:                   2021-02-13T14:41:47Z
    --     Host:                        Linux 5.10.14-051014-generic x86_64
    --     CMake:                       3.16.3
    --     CMake generator:             Unix Makefiles
    --     CMake build tool:            /usr/bin/make
    --     Configuration:               RELEASE
    -- 
    --   CPU/HW features:
    --     Baseline:                    SSE SSE2 SSE3
    --       requested:                 SSE3
    --     Dispatched code generation:  SSE4_1 SSE4_2 FP16 AVX AVX2 AVX512_SKX
    --       requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX
    --       SSE4_1 (16 files):         + SSSE3 SSE4_1
    --       SSE4_2 (2 files):          + SSSE3 SSE4_1 POPCNT SSE4_2
    --       FP16 (1 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 AVX
    --       AVX (6 files):             + SSSE3 SSE4_1 POPCNT SSE4_2 AVX
    --       AVX2 (30 files):           + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2
    --       AVX512_SKX (7 files):      + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2 AVX_512F AVX512_COMMON AVX512_SKX
    -- 
    --   C/C++:
    --     Built as dynamic libs?:      YES
    --     C++11:                       YES
    --     C++ Compiler:                /usr/bin/c++  (ver 9.3.0)
    --     C++ flags (Release):         -fsigned-char -ffast-math -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -O3 -DNDEBUG  -DNDEBUG
    --     C++ flags (Debug):           -fsigned-char -ffast-math -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -g  -O0 -DDEBUG -D_DEBUG
    --     C Compiler:                  /usr/bin/cc
    --     C flags (Release):           -fsigned-char -ffast-math -W -Wall -Werror=return-type -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fopenmp -O3 -DNDEBUG  -DNDEBUG
    --     C flags (Debug):             -fsigned-char -ffast-math -W -Wall -Werror=return-type -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fopenmp -g  -O0 -DDEBUG -D_DEBUG
    --     Linker flags (Release):      -Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a   -Wl,--gc-sections -Wl,--as-needed  
    --     Linker flags (Debug):        -Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a   -Wl,--gc-sections -Wl,--as-needed  
    --     ccache:                      NO
    --     Precompiled headers:         NO
    --     Extra dependencies:          m pthread /usr/lib/x86_64-linux-gnu/libGL.so /usr/lib/x86_64-linux-gnu/libGLU.so cudart_static -lpthread dl rt nppc nppial nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps cublas cufft -L/usr/local/cuda-11.0/lib64 -L/usr/lib/x86_64-linux-gnu
    --     3rdparty dependencies:
    -- 
    --   OpenCV modules:
    --     To be built:                 aruco bgsegm bioinspired calib3d ccalib core cudaarithm cudabgsegm cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev cvv datasets dnn dnn_objdetect dpm face features2d flann freetype fuzzy hdf hfs highgui img_hash imgcodecs imgproc line_descriptor ml objdetect optflow phase_unwrapping photo plot reg rgbd saliency shape stereo stitching structured_light superres surface_matching text tracking ts video videoio videostab xfeatures2d ximgproc xobjdetect xphoto
    --     Disabled:                    cudacodec java_bindings_generator python_bindings_generator world
    --     Disabled by dependency:      -
    --     Unavailable:                 cnn_3dobj java matlab ovis python2 python3 sfm viz
    --     Applications:                tests perf_tests apps
    --     Documentation:               NO
    --     Non-free algorithms:         YES
    -- 
    --   GUI: 
    --     QT:                          YES (ver 5.9.7)
    --       QT OpenGL support:         YES (Qt5::OpenGL 5.9.7)
    --     GTK+:                        NO
    --     OpenGL support:              YES (/usr/lib/x86_64-linux-gnu/libGL.so /usr/lib/x86_64-linux-gnu/libGLU.so)
    --     VTK support:                 NO
    -- 
    --   Media I/O: 
    --     ZLib:                        /usr/lib/x86_64-linux-gnu/libz.so (ver 1.2.11)
    --     JPEG:                        /usr/lib/x86_64-linux-gnu/libjpeg.so (ver 80)
    --     WEBP:                        build (ver encoder: 0x020f)
    --     PNG:                         /usr/lib/x86_64-linux-gnu/libpng.so (ver 1.6.37)
    --     TIFF:                        build (ver 42 - 4.0.10)
    --     JPEG 2000:                   /usr/lib/x86_64-linux-gnu/libjasper.so (ver 1.900.1)
    --     OpenEXR:                     build (ver 2.3.0)
    --     HDR:                         YES
    --     SUNRASTER:                   YES
    --     PXM:                         YES
    -- 
    --   Video I/O:
    --     DC1394:                      YES (ver 2.2.5)
    --     FFMPEG:                      YES
    --       avcodec:                   YES (ver 58.54.100)
    --       avformat:                  YES (ver 58.29.100)
    --       avutil:                    YES (ver 56.31.100)
    --       swscale:                   YES (ver 5.5.100)
    --       avresample:                NO
    --     GStreamer:                   NO
    --     libv4l/libv4l2:              NO
    --     v4l/v4l2:                    linux/videodev2.h
    -- 
    --   Parallel framework:            TBB (ver 2020.1 interface 11101)
    -- 
    --   Trace:                         YES (with Intel ITT)
    -- 
    --   Other third-party libraries:
    --     Intel IPP:                   2020.0.0 Gold [2020.0.0]
    --            at:                   /home/yohann/SLAMs/depend/opencv-3.4.13/build/3rdparty/ippicv/ippicv_lnx/icv
    --     Intel IPP IW:                sources (2020.0.0)
    --               at:                /home/yohann/SLAMs/depend/opencv-3.4.13/build/3rdparty/ippicv/ippicv_lnx/iw
    --     Lapack:                      NO
    --     Eigen:                       NO
    --     Custom HAL:                  NO
    --     Protobuf:                    build (3.5.1)
    -- 
    --   NVIDIA CUDA:                   YES (ver 11.0, CUFFT CUBLAS FAST_MATH)
    --     NVIDIA GPU arch:             80
    --     NVIDIA PTX archs:
    -- 
    --   OpenCL:                        YES (no extra features)
    --     Include path:                /home/yohann/SLAMs/depend/opencv-3.4.13/3rdparty/include/opencl/1.2
    --     Link libraries:              Dynamic load
    -- 
    --   Python (for build):            /home/yohann/anaconda3/bin/python3
    -- 
    --   Install to:                    /usr/local
    -- -----------------------------------------------------------------
    -- 
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /home/yohann/SLAMs/depend/opencv-3.4.13/build

## Problems Encountered
  1. E: Unable to locate package libjasper-dev
  
    sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
    sudo apt update
    sudo apt install libjasper-dev
    
  2. nvcc fatal: Unsupported gpu architecture 'compute_86'
  
   > CUDA outdated, 'compute_86' is supported from CUDA 11.1.1
  
   Sol: 
  
   i) update CUDA from 11.0 to 11.1.1;
  
   ii) set -D CUDA_ARCH_BIN=8.0 when building opencv. (This solution is chosen here)
  
  3. LIBTIFF Error:
  
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFReadRGBAStrip@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFReadDirectory@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFWriteEncodedStrip@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFIsTiled@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFWriteScanline@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFGetField@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFScanlineSize@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFWriteDirectory@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFReadEncodedTile@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFReadRGBATile@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFClose@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFClientOpen@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFRGBAImageOK@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFOpen@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFReadEncodedStrip@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFSetField@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFSetWarningHandler@LIBTIFF_4.0'
    /usr/bin/ld: ../../lib/libopencv_imgcodecs.so.3.4.13: undefined reference to `TIFFSetErrorHandler@LIBTIFF_4.0'
    
   Sol: 在cmake 编译OpenCV时： cmake -D BUILD_TIFF=ON
  
  4. CAIRO Error: 
  
    /usr/bin/ld: /lib/x86_64-linux-gnu/librsvg-2.so.2: undefined reference to `cairo_tag_end'
    /usr/bin/ld: /lib/x86_64-linux-gnu/librsvg-2.so.2: undefined reference to `cairo_tag_begin'
    /usr/bin/ld: /lib/x86_64-linux-gnu/librsvg-2.so.2: undefined reference to `cairo_font_options_get_variations'

   > Anaconda environments caused the mismatch between librsvg and libcairo.
   
   Sol: 
   
   i) set LD_PRELOAD to the path of match librsvg and libcairo;
   
   ii) remove Anaconda and reinstall after opencv is compiled. (This solution is chosen here)
  
  5. python error: no module named 'cv2'
  
    pip install opencv-python

[Back to Top](#table-of-content)


# Anaconda
  Install Anaconda, follow the official instruction [here](https://docs.anaconda.com/anaconda/install/linux/)
  
  *recommend to install OpenCV before Anaconda.*

[Back to Top](#table-of-content)


# NOCS Network
  Paper by Wang et al. in CVPR 2019, [GitHub](https://github.com/hughw19/NOCS_CVPR2019).
  
  Instruction on upgrading code from tf-1 to tf-2 is [here](https://www.tensorflow.org/guide/upgrade).
  
  1. Create anaconda environment and enter virtual environment
    
    conda create -n NOCS2 python=3.8
    conda activate NOCS2
    
  2. Install tensorflow (2.4.0 chosen here)
    
    pip install --upgrade pip
    pip install --upgrade tensorflow-gpu==2.4
    
  3. Install keras (2.4.3 chosen here)
  
    pip install keras
    
  Note: There is not any keras-gpu package; Keras is a wrapper around some backends, including Tensorflow, and these backends may come in different versions, such as tensorflow and tensorflow-gpu. But this does not hold for Keras itself
  
  UPDATE: there is now a keras-gpu package on Anaconda Cloud.

    conda install -c anaconda keras-gpu

   This will install Keras along with both tensorflow and tensorflow-gpu libraries as the backend. (There is also no need to install separately the CUDA runtime and cudnn libraries as they are also included in the package - tested on Windows 10 and working).
   
   4. Some additional packages (scikit-image, open3d, pycocotools):
   
     pip install opencv-python
     pip install scikit-image
     pip install --user --pre https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-0.12.0+a7cfdb2-cp38-cp38-linux_x86_64.whl
     conda install -c conda-forge pycocotools
     
   Change pip to Tsinghua mirror in one installation in case of the super slow installation:
     
     pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
    
## Problems Encountered

  1. CondaHTTPError: 
  
  > HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/linux-64/current_repodata.json>
  
  Sol: Change to Tsinghua Mirror using the instruction [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).
  
  *To change back to original conda source, use*
  
    conda config --remove-key channels

[Back to Top](#table-of-content)


# Libfusion
  Implementation of our SLAM system.

[Back to Top](#table-of-content)
