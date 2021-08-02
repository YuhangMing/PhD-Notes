# PhD-Notes
Some study notes on various areas (after the object based relocalisation project).

## Environment Setup
* [Laptop - Asus TUF Gaming A15](./AsusTUF/README.md)
* [Laptop - Dell XPS 15 9560](./DellXPS/README.md)

## SLAM Related
* [SDF](./SDF/README.md)
* Kalman Filter & Extended Kalman Filter
* Particle Filter

## Semantic for SLAM Related

## Other Learning Related
* [KPConv](./KPConv/README.md)
* [Graph and GCN](./GCN/README.md)
* [Bayesian Filter](https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation): a general probabilistic approach for estimating an unknown probability density function (PDF) recursively over time using incoming measurements and a mathematical process model.
* [Optimiser](./Optimiser/README.md)
  Adam配置参数
  * alpha:   也被称为学习速率或步长。权重比例被校正(例如001)。更大的值(例如0.3)在速率校正之前会加快初始学习速度。较小的值(例如1.0e-5)在培训期间降低学习速度
  * beta1:   第一次估计的指数衰减率(如9)。
  * beta2:   第二次估计的指数衰次减率(例如999)。在稀疏梯度问题(例如NLP和计算机视觉问题)上，这个值应该接近1.0。
  * epsilon: 是一个非常小的数字，可以防止任何在实施中被0划分(例如，10e-8)。
* Traditional ML Algorithms
  * Random Forest
  * AdaBoost
  * ...

## Datasets
### Place Recognition
* [Oxford Robot Car](https://robotcar-dataset.robots.ox.ac.uk/)
* [InHouse Dataset](https://drive.google.com/drive/folders/1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D)
* [InLoc Dataset](http://www.ok.sc.e.titech.ac.jp/INLOC/)
* [Stanford-2D-3D-S](http://buildingparser.stanford.edu/dataset.html)
### 3D Segmentation
* [ScanNet](http://www.scan-net.org/)
* [S3DIS](http://buildingparser.stanford.edu/dataset.html)
* [SUN RGBD Benchmark](https://rgbd.cs.princeton.edu/)
* [NYUv1&v2](https://cs.nyu.edu/~silberman/datasets/)
* [Semantic3D](http://www.semantic3d.net/)
* [Paris-Lille-3D](https://npm3d.fr/paris-lille-3d)
* [SceneNN](http://103.24.77.34/scenenn/home/)
* [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset) (virtual)
* [Scan2CAD](https://github.com/skanti/Scan2CAD)
### RGB-D SLAM
* [ScanNet](http://www.scan-net.org/)
* [SUN RGBD Benchmark](https://rgbd.cs.princeton.edu/)
* [NYUv1&v2](https://cs.nyu.edu/~silberman/datasets/)
* [Seven Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
* [Augmented ICL-NUIM Dataset](http://redwood-data.org/indoor/dataset.html) (virtual)
* [TUM RGBD Benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset)
* [DIML/CVL RGBD Datasest](https://dimlrgbd.github.io/) (both indoors and outdoors)
### Point Clouds
* [3DMatch](https://3dmatch.cs.princeton.edu/)
### More
* [More](http://www.michaelfirman.co.uk/RGBDdatasets/)


## Programming
* [C++](./C++/README.md)
* OpenGL: [CN](https://learnopengl-cn.github.io/01%20Getting%20started/04%20Hello%20Triangle/), [EN](http://www.songho.ca/opengl/index.html).
