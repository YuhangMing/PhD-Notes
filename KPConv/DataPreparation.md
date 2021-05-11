
# SCANNET dataset (ClougSeg)

## Observations
![legends](./imgs/scannet-legends.png)
The test model is trained with only vertical rotation augmentation only.
![Original setup](./imgs/scannet-orig-setup.png)
Results with unchanged input point cloud.
![90RotY setup](./imgs/scannet-90RotY.png)
Results after rotating the input point cloud along y-axis for 90 degrees.

Although the input points coordinates are centered at the selected center point (as said [here](#neighbors-of-center-point)), horizontal planar patches and vertical planar patches can still be separated by looking at the changes in the local coordinates. Then horizontal planar patches are tend to classified as floors (green) and vertical planar patches are tend to classified as walls (blue).


## Containers To Get Network Input
`self.input_trees` stores a list of KD-Trees, generated from SUB-SAMPLED point cloud and used for ..., SUB-SAMPLED points are also sotred in the Trees;
Q: What does the KD-Tree do?

`self.input_colors` stores the RGB values for every 3D points in the SUB-SAMPLED point cloud;

`self.input_vert_inds` stores the map from SUB-SAMPLED point cloud to the ORIGINAL point cloud, i.e. for evey subsampled point, a map stores the index of the corresponding original point;

`self.input_labels`, for Training and Validation only, stores the semantic label for every 3D points in the SUB-SAMPLED point cloud;

`self.pot_trees` stores a list of coarser KD-Trees, generated from SUB-SUB-SAMPLED point cloud and used for ..., SUB-SUB-SAMPLED points are also sotred in the Trees;
Q: What does the coarser KD-Tree do?
A: Used to better find the input center points.

`self.validation_labels` stores the ground truth label of the validation point cloud, in the same dimension as the number of the ORIGINAL point cloud;

`self.test_proj` stores a list of indices which maps every point in the ORIGINAL point cloud to its closet point in the SUB-SAMPLED point cloud. 
`self.test_proj` is NOT exactly the inverse mapping of `self.input_vert_inds`. Some original ponts are mapped to a different subsampled point rather than the one it generates.

Following 3 are all in the same length as the pot_trees:
`self.potentials` stores the full potentials values (each SUB-SUB-SAMPLED point is assigned with a potential number);
`self.min_potentials` stores teh minimum potential values for each pot_tree;
`self.argmin_potentials` stores the index of minimum potential value for each pot tree.

<!-- self.num_clouds = 0 -->

## Network Inputs

Example input list see example_batch_data.txt in dataset folder

[Points](#data-augmentation) coordinates (centered at selected center point) in each layer
```python
# eg. [21604, 3], [5473, 3], [1392, 3], [335, 3], [86, 3]
self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
```
[Neighbors Indices](#convolution-neighbors) (indices in Points array above) in each layer for convolution
```python
# eg. [21604, 63], [5473, 61], [1392, 397], [335, 277], [86, 86]
self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
```
[Pooling/Upsampling Indices](#pooling-upsampling)
```python
# eg. [5473, 58], [1392, 57], [335, 394], [86, 273], [0, 1]
self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
# eg. [21604, 62], [5473, 60], [1392, 281], [335, 86], [0, 1]
self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
```

[Number of Points](#data-augmentation) in each layer
```python
# eg. [21604, 5473, 1392, 335, 86]
self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
```

[Input Features](#input-features)
```python
# dim = [21604, 4], val = [1, r, g, b]
self.features = torch.from_numpy(input_list[ind])
```

[Semantic Labels](#neighbors-of-center-point) of points in the first layer
```python
# [21604]
self.labels = torch.from_numpy(input_list[ind])
```

[Augmentation Parameters](#data-augmentation)
```python
# 3x1 array
self.scales = torch.from_numpy(input_list[ind])
# 3x3 array
self.rots = torch.from_numpy(input_list[ind])
```

[Cloud Index](#get-center-point) in case of multiple clouds inputed
```python
self.cloud_inds = torch.from_numpy(input_list[ind])
```

[Center Point](#get-center-point) index in the coarser KDTree
```python
self.center_inds = torch.from_numpy(input_list[ind])
```

[Input Point Indices](#neighbors-of-center-point) of the input points in the KDTree
```python
# [21604]
self.input_inds = torch.from_numpy(input_list[ind])
```

## General FLow
```
        scene_vh_clean_2.ply
            |
            V
        scene_mesh.ply
      /               |
     /                V
    /                scene_finer_pc.ply
    |                 |                \
    |                 |                |
    |                 V                V
    |    <--   scene_KD_Tree.pkl  scene_sub_pc.ply
    V                 |
scene_proj.pkl        V
               coarse_KD_Tree.pkl
```


## Prepare Polygon Files In DataLoader

#### Get Points and Colors
Get 3D points (Nx3) and color (Nx3) as *numpy ndarray* from *low resolution mesh*;
``` python
vertex_data, faces = read_ply(join(path_to_dataset, scene, scene + '_vh_clean_2.ply'), triangular_mesh=True)
vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T
```


#### Preprocessing before training;

##### Realign and Assign semantic labels: 
1. Realign the 3D points using the *axisAlignment* matrix from dataset folder;
Rotate all points along z-axis and translate all points, example see below. This implies that the z-axis is also the gravity direction?
```
0.233445  0.972370 0.000000 -3.028160 
-0.972370 0.233445 0.000000 2.847190 
0.000000  0.000000 1.000000 -0.106203 
0.000000  0.000000 0.000000 1.000000 
```
2. Get object segmentations of the point cloud;
json file loaded as a dictionary and each point in the vertices is assigned to a segment ID in `segmentations['segIndices']`.
```python
with open(join(path_to_dataset, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
    segmentations = json.load(f)
segIndices = np.array(segmentations['segIndices'])
```
3. Get the instance-level object annotation;
Assign a semantic label to each segments in the object segmentation from previous step.
```python
with open(join(self.ply_path, scene, scene + '.aggregation.json'), 'r') as f:
    aggregation = json.load(f)
```
4. Assign the semantic label to every points in vertices.
```python
for segGroup in aggregation['segGroups']:
    for segment in segGroup['segments']:
        vertices_labels[segIndices == segment] = nyuID
```
5. Save this semantic augmented mesh as a new file
```python
write_ply(join(self.mesh_path, scene + '_mesh.ply'), [vertices, vertices_colors, vertices_labels],
            ['x', 'y', 'z', 'red', 'green', 'blue', 'class'], triangular_faces=faces)
```

##### Get a finer point cloud:
1. Create new points on the mesh faces;
`associated_vert_inds` maps each point in the finer point cloud to a point in the original point cloud
```python
points, associated_vert_inds = rasterize_mesh(vertices, faces, 0.003)
```
2. Grid subsampling on the finer point cloud;
`sub_vert_inds` maps each point in the subsampled point cloud to a point in the original point cloud (before fining)
This **[sub_points, sub_colors, sub_labels, sub_vert_inds]** will be used to generate the **[inputs](#network-inputs)** of the network.
```python
sub_points, sub_vert_inds = grid_subsampling(points, labels=associated_vert_inds, sampleDl=0.01)
sub_colors = vertices_colors[sub_vert_inds.ravel(), :]
sub_labels = vertices_labels[sub_vert_inds.ravel()]
```
3. Save the final fined and then subsampled point cloud and name it Finer_PC;
```python
write_ply(join(path, scene + '.ply'), [sub_points, sub_colors, sub_labels, sub_vert_inds],
            ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])
```


#### Prepare the data for loading.
##### Create KD-Tree:
1. Load the Finer_PC and perform grid subsample;
`points` is the finer point cloud (at dl=0.01),
`colors`, `int_features` are cooresponding rgb values, semantic labels, and corresponidng point index in the original point cloud,
`sub_points`, `sub_colors` are the subsampled points (at dl=0.04) and corresponding colors,
`sub_int_features` maps each point in the subsampled point cloud to a point in the original point cloud, in terms of semantic labels and also the indices.
```python
data = read_ply(file_path)
points = np.vstack((data['x'], data['y'], data['z'])).T
colors = np.vstack((data['red'], data['green'], data['blue'])).T
int_features = np.vstack((data['vert_ind'], data['class'])).T
sub_points, sub_colors, sub_int_features = grid_subsampling(
    points, features=colors, labels=int_features, sampleDl=dl
    )
```
2. Generate KD-Tree (using sklearn.neighbors.KDTree);
```python
search_tree = KDTree(sub_points, leaf_size=10)
```
iii. Store the tree, subsampled point cloud,
```python 
with open(path_to_KDTree, 'wb') as f:
    pickle.dump(search_tree, f)
write_ply(path_to_sub_ply, [sub_points, sub_colors, sub_labels, sub_vert_inds],
            ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])
```
and the [input data containers](#containers-to-get-network-input).
```python
self.input_trees += [search_tree]
self.input_colors += [sub_colors]
self.input_vert_inds += [sub_vert_inds]
self.input_labels += [sub_labels]
```

##### Get Coarse Potential Locations:
1. Subsample the previously subsampled point cloud at a coarser level (e.g. pot_dl=0.2)
```python
sub_points = np.array(self.input_trees[i].data, copy=False)
coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)
search_tree = KDTree(coarse_points, leaf_size=10)
```
2. Save the coarser tree
```python 
with open(path_to_coarser_KDTree, 'wb') as f:
    pickle.dump(search_tree, f)
```
and the [input data containers](#containers-to-get-network-input).
```python
self.pot_trees += [search_tree]
```
##### Get Reprojection Indices (Test and Validalidation ONLY):
Find the map from original point cloud to the subsampled point cloud
1. Load the original point cloud
```python
vertex_data, faces = read_ply(join(self.mesh_path, scene+'_mesh.ply'), triangular_mesh=True)
points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
```
2. For every original point, find its closest neighbour in the sub-sampled point cloud
```python
idxs = self.input_trees[i].query(points, return_distance=False)
```
3. Save the indices
```python 
with open(proj_file, 'wb') as f:
    pickle.dump([proj_inds, labels], f)
```
and the [input data containers](#containers-to-get-network-input).
```python
self.test_proj += [proj_inds]
self.validation_labels += [labels]
```


## \_\_getitem\_\_ in DataLoader
Returns a Batch of [Network Inputs](#network-inputs)

#### Get a Center Point and Update Potentials
##### Get Center Point:
Selected the point with minimum potential across all pot_trees as the center points
`cloud_ind` and `point_ind` are [Network Inputs](#network-inputs)
```python
cloud_ind = int(torch.argmin(self.min_potentials))
point_ind = int(self.argmin_potentials[cloud_ind])
pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)
center_point = pot_points[point_ind, :].reshape(1, -1)
```

##### Update Potentials
1. Search the neighbours in the given raidus
```python
pot_inds, dists = self.pot_trees[cloud_ind].query_radius(
    center_point, r=self.config.in_radius, return_distance=True
    )
```
2. Update potentials using Tukey weights
Larger weights are assigned to the points closer to the center point, 0s are assigned to the points at threshold or larger than the distance threshold.
Meaning, in this way, it tends to find previsouly unused points as new center points.
```python
tukeys = np.square(1 - squared_dists / np.square(self.config.in_radius))
tukeys[squared_dists > np.square(self.config.in_radius)] = 0
self.potentials[cloud_ind][pot_inds] += tukeys
min_ind = torch.argmin(self.potentials[cloud_ind])
self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
self.argmin_potentials[[cloud_ind]] = min_ind
```


#### Get the rest data given the cloud and center point.

##### Neighbors of Center Point:
`input_inds`: indices of the neighbours of this input point in SUB-SAMPLED point cloud ([Network Inputs](#network-inputs)).
`points`: SUB-SAMPLED points,
`input_points`, `input_colors`, `input_labels` are the 3D coordinates (centerd at the selected point), RGB values, semantic labels of the neighbour points ([Network Inputs](#network-inputs)).
```python
input_inds = self.input_trees[cloud_ind].query_radius(center_point, r=self.config.in_radius)[0]

points = np.array(self.input_trees[cloud_ind].data, copy=False)

input_points = (points[input_inds] - center_point).astype(np.float32)
input_colors = self.input_colors[cloud_ind][input_inds]
input_labels = self.input_labels[cloud_ind][input_inds]
input_labels = np.array([self.label_to_idx[l] for l in input_labels])
```

##### Data Augmentation:
Apply a random Rotation, a random Scale factor, and a random Noise to the input points.
`input_points`, `scale`, `R` are [Network Inputs](#network-inputs).
*Currently only vertical augmentation applied (i.e. rotate on the XY plane, around z-axis)*
```python
input_points, scale, R = self.augmentation_transform(input_points)
```
##### Input Features:
`input_features` is [Network Inputs](#network-inputs).
```python
input_features = np.hstack((input_colors, input_points + center_point)).astype(np.float32)
```
Note that not all features here are used. Depending on the `Config.in_features_dim`.
```
if config.in_features_dim == 1:
    [1]
elif config.in_features_dim == 4:
    [1 r g b]
elif config.in_features_dim == 5:
    [1 r g b additional_feats]
else:
    raise ValueError('Error')
```


#### Create the list of network input
[[points](#data-augmentation), [neighbors](#convolution-neighbors), [pools](#pooling-upsampling), [upsamples](#pooling-upsampling), [lengths](#data-augmentation), [features](#input-features), [labels](#neighbors-of-center-point)]

##### Convolution Neighbors:
`conv_i` is [Network Inputs](#network-inputs).
Third-party library used here, returns conv_i is a NxM array, N is the stack length, i.e. num of query points, M is the number of neighbors elements are the indices of the neighbor points with the first element be the query point itself
```python
conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)
```
##### Pooling Upsampling:
`pool_i` and `up_i` are [Network Inputs](#network-inputs).
`pool_p` is the subsampled point cloud, `pool_b` is the length of the subsampled point cloud;
`pool_i` is an N_2 x M_1 array, for each subsample point in `pool_p`, find its M_1 closet neighbor in `stacked_points`;
`up_i` is an N_1 x M_2 array, for each input point in `stacked_points`, find its M_2 closet neighbor in `pool_p`.
```python
pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)
pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)
up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)
```
#### Additional Input for Testing:
[[scales](#data-augmentation), [rots](#data-augmentation), [cloud_inds](#get-center-point), [point_inds](#get-center-point), [input_inds](#neighbors-of-center-point)]



## Setup Batch Sampler
1. Calibration
    1-1. Neibhgor calibrations:
    i. Compute higher bound of the neighbours number in the neighbourhood, the volume of the sphere is used here. Note the radius here is in the number of grid, hence the volume is the total number of grid in the neighborhood.
    E.g. with defrom_radius=6.0, the upper bound of the neighour numbers is 1437.
    ```python
    hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))
    ```