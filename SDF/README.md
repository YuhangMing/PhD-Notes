# Signed-Distance Function
It is an implicit surface representation. Originally used in SLAM by Newcombe *et al.* [KinectFusion](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf).
* Pros: Detailed, dense reconstructon of the environment;
* Cons: Not Scalable due to the pre-allocated grids (memory footprint grows cubically).

# Improved SDF
More efficient data structures have been proposed.
* moving volumes: [Kintinuous](https://github.com/mp3guy/Kintinuous);
* octrees: [FastFusion](https://github.com/tum-vision/fastfusion), [SuperEight](https://github.com/emanuelev/supereight), [Zeng *et al.*](Octree-based fusion for realtime 3D reconstruction);
* $N^3$ trees: [Chen *et al.*](https://people.csail.mit.edu/jiawen/kfhd/kfhd.pdf)
* hash-tables: [InfiniTAM](https://github.com/victorprad/InfiniTAM), [Voxel-Hashing](https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf).