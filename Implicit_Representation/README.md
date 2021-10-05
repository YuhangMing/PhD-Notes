# Table of Content
- [Trending](#trending)
- [NeRF](#nerf)
- [iMAP](#imap)
- [UnsupervisedR&R](#unsupervisedrr)

# Trending
Encoding objects and scenes in the weights of an MLP that directly maps from a 3D spatial location to an implicit representation of the shape.

**What is an implicit representation**:
In mathematics, an implicit surface is a surface in Euclidean space defined by an equation F(x, y, z) = 0. An implicit surface is the set of zeros of a function of three variables. And implicit means that the equation is not solved for x or y or z. [[Wiki](https://en.wikipedia.org/wiki/Implicit_surface)]

**Example of implicit representation of the shape**:
The signed distance at a location. [paper](https://graphics.stanford.edu/papers/volrange/volrange.pdf) [SDF](../SDF/README.md)

**Neural 3D shape representation**:
- Map xyz coordinates to signed distance functions [[DeepSDF](https://github.com/facebookresearch/DeepSDF)], [[Local Implicit Grid](https://github.com/tensorflow/graphics/tree/master/tensorflow_graphics/projects/local_implicit_grid)] or occupancy fields [[Occupancy Network](https://github.com/autonomousvision/occupancy_networks)], [[LDIF](https://github.com/google/ldif)]. GOUND TRUTH 3D GEOMETRY REQUIRED FOR TRAINING. Limited to simple shapes with low geometric complexity.

[Back Top](#table-of-content)


# NeRF
General Idea and how the scene is represented understood.
STILL FUZZY ABOUT HOW THE MATH WORKS IN RENDERING THE SCENE USING NETWORK OUTPUTS (c, σ), as well as why the special design helps.
LOOK INTO SEMINAL RAY TRACING WORKS.

GOAL: new view sythesis.

Main Advantage: It overcomes the prohibitive storage costs of discretized voxel grids when modeling complex scenes at high-resolutions.

Novelty: 
1) Add the additional 2D view dependent appearance to the 3D volumes, making it a 5D radiance field. 

To render the *neural radince field* (NeRF) from a particular viewpoint:
1) march camera rays through the scene to generate a sampled seto fo 3D points;
2) use those points and their corresponding 2D viewing directions as input to the MLP to produce set of colours and densities;
3) use classical volume rendering techinques to accumulate those colours and densities into a 2D image.

By applying gradient descent on minimising error between each observed image and the corresponding rendered view across multiple views, the network is encouraged to predict a coherent model of the scene by assigning high volume densities and accurate colors to the locations that actually contain the scene content.

![Pipeline](./imgs/NeRF.png)

#### Modeling the scene as a neural radiance field
Network Input: 3D location **x** = (x, y, z) and 2D viewing direction (θ, φ) (expressed as 3D cartisan unit vector **d**);
Network Output: Colour **c** = (r, g, b) and volume density σ.
Then the scene is repreesented by the MLP as: F_Θ: (**x**, **d**) -> (**c**, σ) and optimise the weights Θ to map each input to its cooresponding output.

FIRST, the MLP F_Θ processes the input 3D coordinate **x** with 8 FC layers (using ReLU activations and 256 channels per layer), and outputs σ and a 256-dimensional feature vector. 
THEN, this feature vector is concatenated with the camera ray’s viewing direction **d** and passed to one additional FC layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color **c**.

#### Rendering novel views from this representation
Colour of the scene is rendered following [classical volume rendering](https://dl.acm.org/doi/10.1145/964965.808594)
The volume density σ(**x**) can be interpreted as the differential probability of a ray terminating at an infinitesimal particle at location x.
The function T(t) denotes the accumulated transmittance along the ray from tn to t, i.e., the probability that the ray travels from tn to t without hitting any other particle.

#### Optimisation
##### Positional encoding
Mapping the inputs to a higher dimensional space using high frequency functions to enable our MLP to more easily approximate a higher frequency function.
A similar mapping is used in the popular [Transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) architecture, where it is referred to as a positional encoding
##### Hierarchical volume sampling
Simultaneously optimize two networks: one “coarse” and one “fine.

[Back Top](#table-of-content)


# iMAP

With RGB-D input, iMAP uses a MLP to represent the 3D volumetric map and casts the SLAM as a Continual Learning problem. The catastrophic forgetting is alleviated with replaying. Keyframes are selected to store and compress past memories. 

![Pipeline](./imgs/iMAP.png)

#### Implicit Network
A 3D volumetric map is represented using a fully-connected neural network Fθ that maps a 3D coordinate to colour and volume density.

Following [NeRF](#nerf), an MLP with 4 hidden layers of feature size 256 is used. The MLP takes in 3D coordinates **p** = (x, y, z) and outputs colour **c** = (r, g, b) and volulme density ρ. i.e. F_θ(**p**) = (**c**, ρ).

Difference to NeRF, the 2D viewing directions are ignored as the directions are mainly for modelling specularities in generating new views while iMAP is not interested in that in a SLAM setup.

Gaussian positional embedding proposed in Fourier Feature Networks [32] to lift the input 3D coordinate into n-dimensional space

#### Rendering
?? Why the inter-sample distance followed by activation can be used as occupancy probability?????

Given a camera pose, we can render the colour and depth of a pixel by accumulating network queries from samples in a back-projected ray.

Given a camera pose T_WC and a pixel coordinate (u, v):
1. Back-project a normalised viewing direction and transform it into world coordinates: **r** = T_WC K^-1 (u,v) with camera intrinsic matrix K.
2. Take a set of N samples along the ray **p**_i = d_i **r** with corresponding depth values {d_1, ..., d_N}. (following NeRF's sampling strategies.)
3. For each **p**_i, get the network prediction (**c**_i, ρ_i) = F_θ(**p**_i)
4. Transform the volume density into an occupancy probability by Multiplying the inter-sample distance δ_i = d_i+1 - d_i and passing this through activation function o_i = 1 - exp(- ρ_i δ_i)
5. Calculate the ray termination probability at each sample **p**_i as w_i = o_i TT_j=1^i-1(1-o_j)
6. Render the colour and depth as the expectations: D[u,v] = Sum_i=1^N w_i d_i, I[u,v] = Sum_i=1^N w_i **c**_i.
7. Depth variance is calculated as: D_var[u,v] = Sum_i=1^N w_i (D[u,v] - d_i)^2.


#### Optimising
The network weights and camera poses are optimised incrementally w.r.t. a sparse set of actively sampled measurements.

##### Tracking
Optimise the pose of current frame w.r.t. the locked network.

Photometric loss: L1_norm between the rendered and hte measured colours.
e_i^p[u,v] = | I_i[u,v] - I'_i[u,v] | Average over all pixels.

Geometric Loss: e_i^g[u,v] = | D_i[u,v] - D'_i[u,v] | Normalised average over all pixels with the depth variancce as the normalisation factor.

Adam optimiser on the weighted sum L_geom + λ_photo L_photo.

##### Mapping
Jointly optimise the network and the camera poses of selected keyframes, which are incrementally chosen based on information gain.



**An obivious question here is: is there any upper limits on how many keframes can be stored? i.e. can this method handle large scale scenes???**

[Back Top](#table-of-content)


# UnsupervisedR&R



[Back Top](#table-of-content)
