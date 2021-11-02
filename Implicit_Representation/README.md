# Table of Content
- [Trending](#trending)
- [Scene Representation Networks NeurIPS2019](#scene-representation-networks)
- [**NeRF** ECCV2020](#nerf)
- [Object-Centric Neural Scene Rendering arXiv2020](#object-centric-neural-scene-rendering)
- [**Semantic Implicit** 3DV2020](#semantic-implicit)
- [**UnsupervisedR&R** CVPR2021](#unsupervisedrr)
- [NeuralRecon CVPR2021](#neuralrecon)
- [NeuralFusion CVPR2021](#neuralfusion)
- [**Semantic NeRF** ICCV2021](#semantic-nerf)
- [**Object-NeRF** ICCV2021](#object-nerf)
- [iMAP ICCV2021](#imap)
- [Continual Neural Mapping ICCV2021](#continual-neural-mapping)
- [**Neural RGBD Surface Reconstruction** arXiv2021](#neural-rgbd-surface-reconstruction)
- [NeuralBlox 3DV2021](#neuralblox)

# QUESTIONS
1. What does the coordinate-based positional encoding do and how to implement that? See [here](#positional-encoding) for answer.


# Trending
*Encoding objects/scenes in the weights of an MLP that directly maps from a 3D spatial location to objects/scenes properties. This MLP serves as an implicit representation of the object/scene.*

**Difference to the "Code" used in the CodeSLAM and its successors**:
1. CodeSLAM use the fixed-length, compact code between encoder and decoder to represent the input while Implicit Represent uses the weights in an MLP to represent the map.
2. CodeSLAM only represent a single RGB or RGB-D frame while Implicit Representation represent the entire map.

**What is an implicit representation**:
In mathematics, an implicit surface is a surface in Euclidean space defined by an equation F(x, y, z) = 0. An implicit surface is the set of zeros of a function of three variables. And implicit means that the equation is not solved for x or y or z. [[Wiki](https://en.wikipedia.org/wiki/Implicit_surface)]

**Example of implicit representation of the shape**:
The signed distance at a location. [paper](https://graphics.stanford.edu/papers/volrange/volrange.pdf) [SDF](../SDF/README.md)

**Neural 3D shape representation**:
- Map xyz coordinates to signed distance functions [[DeepSDF](https://github.com/facebookresearch/DeepSDF)], [[Local Implicit Grid](https://github.com/tensorflow/graphics/tree/master/tensorflow_graphics/projects/local_implicit_grid)] or occupancy fields [[Occupancy Network](https://github.com/autonomousvision/occupancy_networks)], [[LDIF](https://github.com/google/ldif)]. GOUND TRUTH 3D GEOMETRY REQUIRED FOR TRAINING. Limited to simple shapes with low geometric complexity.

[Back Top](#table-of-content)




# UnsupervisedR&R
An end-to-end unsupervised approach to learn point cloud registration from raw RGB-D video by leveraging differentiable alignment and rendering to enforce photometric and geometric consistency between frames.

The key Idea is that use the natural transformations in the data as indirect supervision provided in the RGB-D video.

The network is trained relying on consistency loss instead of pose supervision.

Limited to small viewpoint-changes (i.e. large overlaps)

![unsupervisedrr](./imgs/unsupervisedrr.png)

Details refer to the  link [here](./UnsupervisedRR.md)

[Back Top](#table-of-content)




# NeRF
[[ref doc](http://www.4k8k.xyz/article/g11d111/118959540)]
GOAL: new view sythesis.

Main Advantage: It overcomes the prohibitive storage costs of discretized voxel grids when modeling complex scenes at high-resolutions.

Novelty: 
1) Add the additional 2D view dependent appearance to the 3D volumes, making it a 5D radiance field. 

To render the *neural radince field* (NeRF) from a particular viewpoint:
1) march camera rays through the scene to generate a sampled seto fo 3D points;
2) use those points and their corresponding 2D viewing directions as input to the MLP to produce set of colours and densities;
3) use classical volume rendering techinques to accumulate those colours and densities into a 2D image.

By applying gradient descent on minimising error between each observed image and the corresponding rendered view across multiple views, the network is encouraged to predict a coherent model of the scene by assigning high volume densities and accurate colours to the locations that actually contain the scene content.

![nerf](./imgs/NeRF.png)

#### Modeling the scene as a neural radiance field
Network Input: 3D location **x** = (x, y, z) and 2D viewing direction (θ, φ) (expressed as 3D cartisan unit vector **d**);
Network Output: Colour **c** = (r, g, b) and volume density σ.
Then the scene is repreesented by the MLP as: F_Θ: (**x**, **d**) -> (**c**, σ) and optimise the weights Θ to map each input to its cooresponding output.

FIRST, the MLP F_Θ processes the input 3D coordinate **x** with 8 FC layers (using ReLU activations and 256 channels per layer), and outputs σ and a 256-dimensional feature vector. 
THEN, this feature vector is concatenated with the camera ray’s viewing direction **d** and passed to one additional FC layer (using a ReLU activation and 128 channels) that output the view-dependent RGB colour **c**.

#### Rendering novel views from this representation
Colour of the scene is rendered following [classical volume rendering](https://dl.acm.org/doi/10.1145/964965.808594)

The expected colour is:

C(**r**) = ∫^{t_f} _{t_n} T(t) σ(**r**(t)) **c**(**r**(t), **d**) dt    - (1)

Explannation:
- C(**r**) is the expected colour at a given camera ray (pixel).
- t is the distance traveled along a camera ray with t_f being the far bound and t_n being the near bound.
- T(t) = exp(−∫^t _{t_n} σ(**r**(s)) ds) denotes the accumulated transmittance along the ray from t_n to t, i.e., the probability that the ray travels from t_n to t without hitting any other particle.
- **r**(t) = **o** + t**d** is the camera ray with **o** being the camera centre and **d** being the 3D direction.
- σ(**r**(t)) is the volume density at location **r**(t) predicted by the MLP. It can be interpreted as the differential probability of a ray terminating at an infinitesimal particle at location **r**(t).
- **c**(**r**(t), **d**) is the view-dependt colour value predicted by the MLP.

Equation (1) is numerically solved using **quadrature** (Riemann Sum). To maintain the resolution of this representation, the stratified sampling approach is adopted where we partition [t_n, t_f] into N evenly-spaced bins and then draw one sample uniformly at random from within each bin.
t_i ∼ U[ t_n + (i-1)/N (t_f − t_n), t_n + i/N (t_f − t_n) ]     - (2)

Then we have the numerical estimation of C(**r**) from the set of (**c**i, σi) values (network outputs):

\hat(C)(**r**) = ∑^N_{i=1} T_i (1 − exp(− σ_i δ_i)) **c**_i     - (3)

where δ_i = t_{i+1} − t_i is the distance between adjacent samples and T_i = exp(- ∑^{i-1}_{j=1}) σ_j δ_j).

!!!? ??? Equation (3) is trivical differentiable ??? reduces to traditional alpha compositing with alpha values αi = 1−exp(−σiδi). ???


#### Optimisation
The following two improvements are introduced to representing high-resolution complex scenes.

##### Positional encoding
Why doing so? The authors found that having the network FΘ directly operate on xyzθφ input coordinates results in renderings that perform poorly at representing high-frequency variation in color and geometry.

Improvements: Mapping the inputs to a higher dimensional space using high frequency functions to enable our MLP to more easily approximate a higher frequency function.
(high-frequency refer to the edges-like region in the scene)

This is done by reformulating F_Θ as a composition of two functions F_Θ = F'_Θ ◦ γ. 

F'_Θ is simply a regular MLP (to be learned). 

γ is applied separately to each of the three coordinate values in **x** (which are normalized to lie in [−1, 1]) and to the three components of the Cartesian viewing direction unit vector **d** (which by construction lie in [−1, 1]). 
In the experiments, the authors set L = 10 for γ(**x**) and L = 4 for γ(**d**).
It is defined as:

γ(p) = (sin(2^0 πp), cos(2^0 πp)), ..., (sin(2^{L−1} πp), cos(2^{L−1}πp))       - (4)

A similar mapping is used in the popular [Transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) architecture, where it is referred to as a *positional encoding*.

Difference: 

- Transformers use it for a different goal of providing the discrete positions of tokens in a sequence as input to an architecture that does not contain any notion of order. 
- NeRF use these functions to map continuous input coordinates into a higher dimensional space to enable our MLP to more easily approximate a higher frequency function.


##### Hierarchical volume sampling
Why doing so?  Current rendering strategy of densely evaluating the neural radiance field network at N query points along each camera ray is inefficient. Because free space and occluded regions that do not contribute to the rendered image are still sampled repeatedly.

Improvement: increases rendering efficiency by allocating samples proportionally to their expected effect on the final rendering.

Main idea: This procedure allocates more samples to regions we expect to contain visible content. i.e. **sample more points around the surface**.

Simultaneously optimize two networks: one "coarse" and one "fine":

The "coarse" network is evaluated at N_c *stratified sampled* locations using Equations (2) and (3).

Given output from the "coarse" network, a new set of N_f locations are sample from the per-ray PDF.

\hat{C}_f(**r**) = ∑^{N_c}_{i=1} \hat{w}_i c_i.
where \hat{w}_i = w_i/∑^{N_c}_{j=1} w_j, w_i = T_i (1 − exp(− σ_i δ_i))

*Limitation raised by [[Object-Centric Neural Scene Rendering](#object-centric-neural-scene-rendering)]: only model static scenes and are closely tied to specific imaging conditions, making NeRFs hard to generalise to new scenarios.*

##### Implementation details
Data: Inputs to the network are 5D vectors. These inputs are computed from SfM using a set of RGB images of the scene, corresponding camera poses, intrinsic parameters, and scene bounds (we use ground truth camera poses, intrinsics, and bounds for synthetic data, and use the COLMAP structure-from-motion package to estimate these parameters for real data).

Our loss is simply the total squared error between the rendered and true pixel colors for both the coarse and fine renderings:

L = ∑_{r∈R} || \hat{C}_c(**r**) − C(**r**) ||^2_2 + || \hat{C}_f(**r**) − C(**r**) ||^2_2

where R is the set of rays in each batch, and C(**r**), \hat{C}_c(**r**), and \hat{C}_f(**r**) are the ground truth, coarse volume predicted, and fine volume predicted RGB colors for ray **r** respectively.

In the experiments, the authors use a batch size of 4096 rays, each sampled at N_c = 64 coordinates in the coarse volume and N_f = 128 additional coordinates in the fine volume. They use the Adam optimizer with a learning rate that begins at 5 × 10^{−4} and decays exponentially to 5 × 10^{−5} over the course of optimization (other Adam hyperparameters are left at default values of β1 = 0.9, β2 = 0.999, and epsilon = 10^{−7}). The optimization for a single scene typically take around 100–300k iterations to converge on a single NVIDIA V100 GPU (about 1–2 days).

[Back Top](#table-of-content)




# Semantic Implicit

Build on Scene Representation Networks ([SRN](#scene-representation-networks)) and perform per-point semantic segmentation in addition to represent appearance and geometry.
The model proposed in this paper is a **Semantically Aware Implicit Neural Scene Representation**.

<u>Good review on representing scenes in the weights of a MLP (compensation to the papers mentioned here). </u>

Small set of semantically labelled data is needed for training.

## Methodology

![semantic_srn](./imgs/Semantic_SRN.png)

#### SRN

*SRN*: Encode a scene in the weights **w** \in R^l of a MLP. Map a 3D coordinate **x** to a scene property **v**.

*RGB Renderer*: 1) use a differentiable ray marcher to find the intersections of camera rays withscene geometry; 2) query the SRN at the intersection points and map the feature vector **v** to an RGB colour using another MLP.

*Hypernetwork*: maps the embedding vector **z** \in R^k to the weight **w** \in R^j, enabling representing an object class using an embedding vector **z**.

#### Semantic

Minimum supervision wanted, authors concludes that the learned features **v** already contains the semantic info (see t-SNE plot).

*Segmentation Render SEG*: Maps a feature vector **v** to a distribution over class labels **y**.

SEG is added to the SRN **in parrallel to** the RGB Renderer and is parameterised as **a linear classifier** with input being the feature vector **v** from SRN and output being class labels.

[Back Top](#table-of-content)




# Semantic-NeRF
### In-Place Scene Labelling and Understanding

Advantages of using implicit neural reconstructions is that they do not require prior training data. But the fully self-supervised approach is not possible for semantics (labels are human-defined).

*Main achievement*: **extend [NeRF](#nerf) to jointly encode semantics with appearance and geometry**.

*Goal*: focused on solving semantic segmentation with sparse and very noisy labels, with applications in **Visual Semantic Mapping Systems** like scene labelling, novel semantic view synthesis, label interpolation, multi-vew semantic label fusion.

Therefore, a **Scene-Specific** network is designed for <u> joint geometric and semantic prediction </u> and train it on images from a **Single Scene** with only *weak semantic supervision and no geometric supervision*. 

<u> Good review on comparing code-based representations with the implicit 3D representations. </u>

![semantic-nerf](./imgs/Semantic-NeRF.png)

Input: a set of RGB images with associated known camera poses and some partial/noisy semantic labels.
Output: implicit 3D representations of both geometry and semantics for the whole scene.

**Semantic added with an additional output branch for segmentation prediction**
*Semantic Segmentation Renderer*: appended to the NeRF before the injection of viewing directions. It is formalised as an inherently view-independent function that maps only a world coordinate **x** to a distribution over C semantic classes via softmax. 
The estimated semantic label given 3D points from a ray is very similiar to the formulation of colour estimation.

The training loss for the entire network is a weighted sum of photometric loss L_p and semantic loss L_s.

The network is trained for **each scene individually**, i.e. train and tested on the same sequences but different frames.

Training images (colour, depth, semantic) are generated using [Habitat](../Tools/README.md)

[Back Top](#table-of-content)




# Object-NeRF

Goal: editable scene rendering.
Drawback of NeRF: Encode the entire scene as a whole and is not aware of the object identity.
**Improvement**: represent background and objects with two separate MLPs. 
Input: posed images and 2D instance masks. 

<u> Good review on object decomposite rendering. </u>

![object-nerf](./imgs/Object-NeRF.png)

Input: 
Hybrid space embedding: apply positional endocing γ(·) ([NeRF](#nerf)) on both of the scene voxel feature **f**_{scn} and the space coordinate **x**.
Direction embedding: γ(**d**).
Input only to the object brance: embedde object voxel features: γ(**f**_{obj}) and object activation code: **l**_{obj}.
* **f**_{obj} helps to broaden the ability of learning decomposition and is shared by all the objects and **l**_{obj} identifies feature space for different objects and is possessedby each individual. *

Output:
Scene branch: the opacity σ_{scn} and colour **c**_{scn} of the scene at **x**.
Object branch: color **c**_{obj} and opacity **σ**_{obj} for the desired object while everything elseremains empty.

Q: what is the activation code and how to get the voxel features. Answered below.

Object-composition:
Supervision is achieved with 2D instance segmentation and assigning a bunch of shuffled object activation codes to the training rays.
Assuming there are K annotated objects in the scene, create a learnable object code library L={**l**_{obj}^k}.
For each ray **r**, select one object k as a training target and assign the object activation code **l**_{obj}^k to the object branch input.
Colour output **c**_{obj} and opacity output **σ**_{obj} is computed in the same way as colour and opacity in NeRF.
The loss of object supervision considers the distance between instance masks and also the distance between the rendered colour and the masked ground-truth colour.

[Back Top](#table-of-content)




# Neural RGBD Surface Reconstruction

**Reference to network structure here**

TL;DR: Replacing the mapping module in BundleFusion with the network and optimisation proposed in this paper.

Leverages the success of implicit novel view synthesis ([NeRF](#nerf)) for surface reconstruction.
Incorporate **depth measurement** into the radiance field formulation to produce mroe detailed and complete reconstruction.

Again, use a deep neural network to store the TSDF. And the beneficial is the same as previous ones, i.e. handling regions with missing depth measurements.

<u> Good review on representing scenes in the weights of a MLP (compensation to the papers mentioned here). </u>

General Steps:
1. Initialisation: obtain camera pose using BundleFusion.
1. Optimisation: optimise a continuous volulmetric representation of the scene that stores radiance and TSDF per point.
1. Evaluation: use Marching Cubes to extract a triangle mesh.

![neural-surface-recon](./imgs/neural-surface-recon.png)

MLP-1:  
Input:  encoding (represented by γ(-)) of a queried 3D point;
Output: the truncated signed distance D_i to the nearest surface (TSDF value).
MLP-2: 
produce surface colour valules for a given viewing direction d.
Input: concatenation of 1) positional encoding of the viewing direction γ(d) (enables dealing with view-dependent effects like specular highlights); 2) a 2-D appearance latent code (learned for each frame following [NeRF in the Wild](https://nerf-w.github.io/) to correct for effects like auto-white balancing); 3) the output of MLP-1.
Output: colour value of the given pixel.

[Back Top](#table-of-content)




# iMAP

With RGB-D input, iMAP uses a MLP to represent the 3D volumetric map and casts the SLAM as a **Continual Learning** problem. The catastrophic forgetting is alleviated with replaying. Keyframes are selected to store and compress past memories. 

![imap](./imgs/iMAP.png)

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

*Limitation raised by [[NeuralBlox](#neuralblox)]: iMAP has limited scalability as the entire scene is represented in one single code*

[Back Top](#table-of-content)




# Continual Neural Mapping

GOAL: continual learn the implicit scene representation directly from sequential observations.
Bridging the gap between batch-trained implicit neural representations and streaming data.

Casted as **Continual Learning** problem, as [iMap](#imap).

## Problem Statement
**y** = f(**x**, θ^t), \all **x** \in W. The goal is to learn the mapping function f(-) parameterised by a neural network θ^t. W is the 3D environment, **x**^t \in Ω^t \subset W is the 3D coordinate at time t, and **y** is the scene property.

**Knowledge Transfer**
- Backward Transfer: [[Continual Learning for Robotics](https://arxiv.org/pdf/1907.00182.pdf)] [[Gradient Episodic Memory](https://arxiv.org/pdf/1706.08840.pdf)]
For unvisited areas, the mapping function f(-) can be queried at any time to predict hte scene property **y** given the spatial cooridinate **x**.
For previsouly visited area **x** \in Ω^{1:t}, the mapping funciton f(-) serves as a compact memory of past observations D^{1:t}.
- Forward Transfer:
May be facilitated that distills knowlege and skills for *future exploration*.

**Learning Paradigms**
Domain-incremental continual learning: data distribution shifts and the objective remains the same.
![learning-paradigms](./imgs/learning_paradigms.png)
- Multi-task learning: splits the training process into a set of dependent tasks and optimizes all tasks jointly. The network is fixed once the model is deployed.
- Fine-tuning: maintains a single network consecutively, where network parameters of a new task are initialized with that of the last task. The performance of early tasks will degrade on current network parameters (*Catastrophic Forgetting*).
- Batch-retraining: preserves all previously observed data to satisfy the iid-sampled assumption. It is computationally expensive as it learns a new model at each time from scratch without exploiting past experience.

[Back Top](#table-of-content)




# Object-Centric Neural Scene Rendering

Proposed to learn object-centric neural scattering functions (OSFs) to model **per-object** light transport.

The entire rendering problem is decomposed into 2 components:
1. a learned component (per-object asset creation) which models intra-object light transport.
2. a non-learned component (per-scene path tracing) which handles inter-object light transport.

#### Background
!!! I can understand what these equations mean but don't really understand why these equations work !!!

##### Volume Rendering
(Similar definition as [NeRF](#nerf))
It's an approach for computing the radiance traveling along rays traced in a volume. **r**(t) = **x**_0 + t**ω**_**o** defines a ponit along a ray **r** with origin **x**_0. Following the Monte Carlo path tracing formulation, the radiance of the ray can be computed as:

L(**x**_0,**ω**_**o**) =∫^t_{f}_t_{n} τ(t) σ(**r**(t)) L_S(**r**(t),**ω**_**o**) dt    - (1)

t_n and t_f are near and far integration bounds.
σ(**r**(t)) denotes the volume density of point **r**(t). 
τ(t) is computed as exp(−∫^t_{t_n} σ(**r**(u))du), which denotes the accumulated transmittance from t_n to t. 
L_S(**r**(t),**ω**_**o**) is the amount of light scattered at point **r**(t) along direction **ω**_**o**, defined as the integral over all incoming light directions:

L_S(**x**,**ω**_**o**) =∫_S L(**x**,**ω**_**l**) f_p(**x**,**ω**_**l**,**ω**_**o**) d**ω**_**l**    - (2)

where S is a unit sphere and f_p is a phase function that evaluates the fraction of light incoming from direction **ω**_**l** at a point **x** that scatters out in direction **ω**_**o**. 

*NeRF assumes fixed illumination and does not consider Equation (2) at all.*

##### Raytracing
According to [[Max TVCG1995](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)] [[Kniss et al. TVCG2003](http://www.sci.utah.edu/publications/jmk03/kniss_tvcg03_volshade.pdf)], quadrature can be used to numerically estimate the integrals in Equation (1).

For each ray stratified sample N samples along the ray, we have the approximation:

L(**x**_0,**ω**_**o**) = ∑^N_{i=1} τ_i α_i L_S(**x**_i,**ω**_**o**)    - (3)

where τ_i = ∏^{i−1}_{j=1} (1−α_j) and α_i = 1−e^{−σ_i (t_{i+1} − t_i)}.
Similarly, the domain S in Equation (2) is also discretised by sampling a set of incoming light paths, leading to the approximation:

L_S(**x**_i,**ω**_**o**) = (1/|L|) ∑_{l∈L} L(**x**_i,**ω**_**l**) **ρ**^**l**_i    - (4)

where **ρ**^**l**_i = f_p(**x**_i,**ω**_**l**,**ω**_**o**) ∈ [0, 1] is the fraction of light incoming from light path **l** that is scattered in direction **ω**_**o**.

##### [NeRF](#nerf)

#### OSFs
![OSFs](./imgs/OSFs.png)

Learn an implicit function F_Θ: (**x**, **ω**_**l**, **ω**_**o**) → (σ, **ρ**).  
Θ are learned weights that parameterize the neural network.
Input: 3D point in the object coordinate frame **x** = (x,y,z), the incoming light direction **ω**_**l** = (φ_i,θ_i), and the outgoing light direction **ω**_**o** = (φ_o,θ_o);
Output: volumetric density σ and fraction of incoming light that is scattered in the outgoing direction **ρ** = (ρ_r,ρ_g,ρ_b).

Positional encoding is also applied to the inputs. And objects are transformed to their canonical coordinate frame.

Question: does this per-object NeRF use a single MLP for ALL objects or use a MLP for EACH object?????
One OSF for EACH object.
An eight-layer MLP with 256 channels to  predict σ, and a four-layer MLP with 128 channels to predict **ρ**.

#### Rendering Multiple OSFs
![Rendering](./imgs/Render_OSFs.png)

Details skipped for now.


[Back Top](#table-of-content)




# NeuralRecon

GOAL: Directly reconstruct local surfaces represented as sparse TSDF volumes for each video fragment.
**Gated Recurrent Units** are used here for the learning-based TSDF fusion module. 
*This work fomulated the SLAM problem differently to [iMAP](#imap) or [Continual Neural Mapping](#continual-neural-mapping). Instead of using the Continual Learning Formulation with a single MLP, NerualRecon adops a CNN-RNN-MLP formulation.*

Input: Monocular images with corresponding camera poses.
Unprojects the image features to form a 3D feature volume and then uses sparse convolutions to process the feature volume to output a sparse TSDF volume.
GRU makes ccurrent reconstruction conditioned on previous global volume.
Speed: 33 keyframes per second on an NVIDIA RTX 2080Ti GPU.

![neural-recon](./imgs/neuralrecon.png)
- Keyframe images passed through iamge backbone to extract multi-level features;
- Backproject the features and aggregated them into a 3D feature volume F_t^l;
- 3D feature volume F_t^l is passed through the GRU and MLP modules to get the predicted Sparse TSDF (Final Output).

[Back Top](#table-of-content)




# NeuralFusion

Input: a stream of depth maps with know camera calibration.
Output: a TSDF map that fuses all surface information while removing noise and outlier and complete potentially missing observations.
Key Idea: decouple the scene representation fro geometry fusion from the output scene representation. This is done by perfusion fusion in the latent feature space and use a translator network to get the final scene representation given the fused global features.

![neuralfusion](./imgs/NeuralFusion.png)
![neuralfusion-arch](./imgs/neuralfusion-arch.png)

[Back Top](#table-of-content)




# Scene Representation Networks

A less direct neural 3D representation: 1) outputs a feature vector and RGB colour at each continueous 3D coordinates; 2) proposes a differentiable rendering function consisting of a recurrent neural network that marches along each ray and decide where the surface is located.

![srn](./imgs/SRN.png)

[Back Top](#table-of-content)




# NeuralBlox

**This work feels like more related to CodeSLAM style rather than NeRF**

Incrementally build and update the neural implicit representations.
The scene of arbitrary size is represented as a dynamically growing grid of voxels with latent codes in them and updates are performed directly in the latent space.
Real-time performance on CPU.

Input: a sequence of point clouds (generated by a LiDAR or a RGB-D camera).
Output: an occupancy map which contains information about free and occupied space.

![neuralblox](./imgs/NeuralBlox.png)

Separated training pipeline:
1. train the encoder and decoder in a supervised way using ShapeNet and synthetic object-level dataset;
1. train another fusion network in a self-supervised manner given the trained encoder and decoder.

#### Local Geometry Representation
The network follows the architecture of [3D grid Convolutional Occupancy Network](https://github.com/autonomousvision/convolutional_occupancy_networks) with input point clouds encoded using [PointNet](https://github.com/charlesq34/pointnet) and [3D U-Net](https://arxiv.org/pdf/1606.06650.pdf). Note that the skip connections are removed from the 3D U-Net.

#### Latent Code Fusion
Sum over all the latent code assigned to one voxel and divided by the counts. This averaged latent code is used as the input to the fusion network.
The fusion network consists of 2 3D convolution layers with ReLU activations.

#### Occupancy Map Generation
The decoder predicts the occupancy probability and threshold is set to determine if a voxel is occupied or free.

[Back Top](#table-of-content)



