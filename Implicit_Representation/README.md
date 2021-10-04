# Trending
Encoding objects and scenes in the weights of an MLP that directly maps from a 3D spatial location to an implicit representation of the shape.

**What is an implicit representation**:
In mathematics, an implicit surface is a surface in Euclidean space defined by an equation F(x, y, z) = 0. An implicit surface is the set of zeros of a function of three variables. And implicit means that the equation is not solved for x or y or z. [[Wiki](https://en.wikipedia.org/wiki/Implicit_surface)]

**Example of implicit representation of the shape**:
The signed distance at a location. [paper](https://graphics.stanford.edu/papers/volrange/volrange.pdf) [SDF](../SDF/README.md)

**Neural 3D shape representation**:
- Map xyz coordinates to signed distance functions [[DeepSDF](https://github.com/facebookresearch/DeepSDF)], [[Local Implicit Grid](https://github.com/tensorflow/graphics/tree/master/tensorflow_graphics/projects/local_implicit_grid)] or occupancy fields [[Occupancy Network](https://github.com/autonomousvision/occupancy_networks)], [[LDIF](https://github.com/google/ldif)]. GOUND TRUTH 3D GEOMETRY REQUIRED FOR TRAINING. Limited to simple shapes with low geometric complexity.


# NeRF

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

Network Input: 3D location **x** = (x, y, z) and 2D viewing direction (θ, φ) (expressed as 3D cartisan unit vector **d**);
Network Output: Colour **c** = (r, g, b) and volume density σ.
Then the scene is repreesented by the MLP as: F_Θ: (**x**, **d**) -> (**c**, σ) and optimise the weights Θ to map each input to its cooresponding output.

FIRST, the MLP F_Θ processes the input 3D coordinate **x** with 8 FC layers (using ReLU activations and 256 channels per layer), and outputs σ and a 256-dimensional feature vector. 
THEN, this feature vector is concatenated with the camera ray’s viewing direction **d** and passed to one additional FC layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color **c**.

KEEP READING FROM SECTION 4.