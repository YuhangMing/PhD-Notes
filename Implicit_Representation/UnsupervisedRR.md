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
[implementation](#data-flow-in-a-forward-pass)
I \in R^{4xHxW} -> P \in R^{(6+F)xN}.
4 channels in the input image are R, G, B, D;
p \in P is represented by a 3D coordinate **x**_p \in R^3, a colour **c**_p \in R^3, and a feature vector **f**_p \in R^F.

3D points are generated with back-project and pixels with missing depth measurements are omitted.

Features are extracted using an encoder, and the feature map has the same spatial resolution as the input image.

#### Correspondences Estimation
[implementation](#data-flow-in-a-forward-pass)
Consice distance is used to determine the closest features. Leading to two sets of correspondences C_{P->Q} and C_{Q->P}.
To estimate a weight for each correspondence, Lowe's ratio test is applied. Computed as the distance between p to its 1st nearest neighbour q_p1 over the distance between p to its 2nd nearest neighbour q_p2.
Final correspondence set is M = {(p, q, w)_i: 0 <= i < k} where k = 400.

#### Geometric Fitting
[implementation](#data-flow-in-a-forward-pass)
Given the set of correspondences M, solve for the optimal transformation T* over an error function. Solved with a weighted variant of Kabsch's algorithm.
A simplified version of RANSAC is also applied to mitigate the problem of outliers. Details refer to [Choy et al. CVPR2020](https://github.com/chrischoy/DeepGlobalRegistration) and [Kabsch's Algorithm ](https://onlinelibrary.wiley.com/doi/abs/10.1107/S0567739476001873?sentby=iucr).

#### Point Cloud Rendering
[implementation](#data-flow-in-a-forward-pass)
Render the RGB-D images from the aligned point clouds serving a verificaiton step.
If the camera locations are estimated correctly, the point cloud renders will be consistent with the input images

#### Possible Improvement
Work on large viewpoint changes

[[Back to Top]](#table-of-content)


## Network Architecture
The model consists of three main components: an *Encoder*, a *Decoder* and a *Renderer*.

### Data flow in a forward pass
Use the *Encoder* to get features for each input view:
```python
feats = [self.encode(rgbs[i]) for i in range(n_views)]
```

Generate the point clouds from an evenlly spaced grid, given RGB-D inputs [link](#point-cloud-generation):
```python
B, _, H, W = feats[0].shape
assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
grid = get_grid(B, H, W)
grid = grid.to(deps[0])

K_inv = K.inverse()
pointclouds = [
    grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
]
pcs_X = [pc[0] for pc in pointclouds]
pcs_F = [pc[1] for pc in pointclouds]
```

Get the correspondences given the features from views using kNN according to a distance metric (default cosine distance) [link](#correspondences-estimation):
```python
corr_i = get_correspondences(
    P1=pcs_F[0], P2=pcs_F[i], P1_X=pcs_X[0], P2_X=pcs_X[i],
    num_corres=self.num_corres, ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
)
```

Align the two point clouds given estimated correspondences by randomly choose N subsets and selects the one that minimises the chamfer distance [link](#geometric-fitting):
```python
Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)
```

For every view, get the point clouds, features and RGBs from all other views (in world coordinate system), then render the joint point cloud with colour and feature [link](#point-cloud-rendering):
```python
for i in range(n_views):
    if self.pointcloud_source == "other":
        # get joint for all values except the one
        pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1 : n_views], dim=1)
        pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1 : n_views], dim=1)
        pcs_RGB_joint = torch.cat(
            pcs_rgb[0:i] + pcs_rgb[i + 1 : n_views], dim=1
        )
        pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

    if i > 0:
        rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])  # transform back to world coordinate system
        rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
    else:
        rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
    projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))
```

Decode the features to get a decoded RGB:
```python
for i in range(n_views):
    proj_FRGB_i = projs[i]["feats"]
    proj_RGB_i = proj_FRGB_i[:, -3:]
    proj_F_i = proj_FRGB_i[:, :-3]

    output[f"rgb_decode_{i}"] = self.decode(proj_F_i)
```

Output summary:
```python
# estimated correspondences
output[f"corres_0{i}"] = corr_i
output[f"vp_{i}"] = Rt_i
output["corr_loss"] = sum(cor_loss)
# aligned pcd
pcs_X_rot = [
    transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True) for i in range(n_views - 1)
]
pcs_X = pcs_X[0:1] + pcs_X_rot
output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()
# decode
output[f"rgb_decode_{i}"] = self.decode(proj_F_i)
# render
output[f"rgb_render_{i}"] = proj_RGB_i
output[f"ras_depth_{i}"] = projs[i]["depth"]
output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)
```

#### Loss
*Loss = Appearance_Loss + Depth_Loss + Correspondence_Loss*
where appearance_loss and depth_loss are calculated between the input view and the rendered view while the correspondence_loss is computed in the kNN stage.



[[Back to Top]](#table-of-content)
