# UnsupervisedR&R

## Table of Content
- [Overview](#conceptual-overivew)

## Conceptual Overview 
An end-to-end unsupervised approach to learn point cloud registration from raw RGB-D video by leveraging differentiable alignment and rendering to enforce photometric and geometric consistency between frames.

Key Idea: use the natural transformations in the data as indirect supervision provided in the RGB-D video.

Trained relying on consistency loss instead of pose supervision.

General Approach:
1. first extract 2D features for each image and project them into two feature point clouds; 
2. extract correspondences between the two point clouds and rank the correspondences based on their uniqueness.
3. use a differentiable optimizer to align the top k correspondences and estimate the 6-DOF transformation between them.
4. render the point cloud from the two estimated viewpoints to generate an RGB image for each view; and use photometric and geometric consistency losses between the RGB-D inputs and outputs and back-propagate through our entire pipeline.

![unsupervisedrr](./imgs/unsupervisedrr.png)

#### Point Cloud Generation
I \in R^{4xHxW} -> P \in R^{(6+F)xN}.
4 channels in the input image are R, G, B, D;
p \in P is represented by a 3D coordinate **x**_p \in R^3, a colour **c**_p \in R^3, and a feature vector **f**_p \in R^F.

3D points are generated with back-project and pixels with missing depth measurements are omitted.

Features are extracted using an encoder, and the feature map has the same spatial resolution as the input image.

#### Correspondences Estimation
Consice distance is used to determine the closest features. Leading to two sets of correspondences C_{P->Q} and C_{Q->P}.
To estimate a weight for each correspondence, Lowe's ratio test is applied. Computed as the distance between p to its 1st nearest neighbour q_p1 over the distance between p to its 2nd nearest neighbour q_p2.
Final correspondence set is M = {(p, q, w)_i: 0 <= i < k} where k = 400.

#### Geometric Fitting
Given the set of correspondences M, solve for the optimal transformation T* over an error function. Solved with a weighted variant of Kabsch's algorithm.
A simplified version of RANSAC is also applied to mitigate the problem of outliers.

#### Point Cloud Rendering
Render the RGB-D images from the aligned point clouds serving a verificaiton step.
If the camera locations are estimated correctly, the point cloud renders will be consistent with the input images

#### Possible Improvement
Work on large viewpoint changes

[[Back to Top]](#table-of-content)


##
