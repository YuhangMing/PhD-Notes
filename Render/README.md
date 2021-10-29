# Reading notes [SoTA on Neural Rendering](https://arxiv.org/abs/2004.03805)

#### Neural Rendering
It is a new and rapidly emerging field that combines **generative machine learning techniques** with **physical knowledge from computer graphics**, e.g., by the integration of differentiable rendering into network training.

Definition: Deep image or video generation approaches that enable explicit or implicit control ofscene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure.
This

While classical computer graphics starts from the perspective of physics, by modeling for example geometry, surface properties and cameras, 
machine learning comes from a statistical perspec- tive, i.e., learning from real world examples to generate new images.

# Theoretical Background

**Light transport**: the physical process that light sources emit photons that interact with the objects in the scene, as a function of their geometry and material properties, before being recorded by a camera.
Light transport considers all the possible paths of light from the emitting light sources, through a scene, and onto a camera.

[UCSD CSE252A]
L_o(**p**, ω_o, λ, t) = L_e(**p**, ω_o, λ, t) + L_r(**p**, ω_o, λ, t)
- L_o represents outgoing radiance from a surface as a function of location, ray direction, wavelength, and time. 
- L_e represents direct surface emission
- L_r represents the interaction of incident light with surface reflectance definded as:
L_r(**p**, ω_o, λ, t) = ∫_Ω f_r(**p**, ω_i, ω_o, λ, t) L_i(**p**, ω_i, λ, t) (ω_i · **n**) dω_i

!!![] is this the equation used in NeRF?

Because the intergral cannot be solved in colsed form for nontrivial scens, approximation must be employed. The most accurate approximations employ Monte Carlo simulations, sampling ray paths through a scene.

**Scene representation**:
*Explicit* methods describe scenes as a collection of geometric primitives, such as triangles, point-like primitives, or higher-order parametric surfaces.
*Implicit* representations include signed distance functions mapping from R^3 → R, such that the surface is defined as the zero-crossing of the function (or any other level-set).
<u>In practice, most hardware and software renderers are tuned to work best on triangle meshes, and will convert other representations into triangles for rendering.</u>
Conversion from implicit representation to explicit representation can be done with *<u>[Marching Cubes](./MarchingCubes.md)</u>* algorithm or similar methods.
*Material properties*: Skipped.

**Camera model**:
Pin-hole camera.

**Classical Rendering**:
The process of transforming a scene definition including cameras, lights, surface geometry and material into a simulated camera image is known as *rendering*.
The process of estimating the different model parameters (camera, geometry, material, light parameters) from real-world data, for the purpose of generating novel views, editing materials or illumination, or creating new animations is known as *inverse rendering*.

*[Rasterization](./Rasterization.md)*: 
- a feed-forward process in which geometry is transformed into the image domain, sometimes in back-to-front order known as painter’s algorithm.
- hardware-accelerated rendering because the good memory coherence.
- requires an explicit geometric representation.

*[Raytracing](./Raytracing.md)*: 
- a process in which rays are cast backwards from the image pixels into a virtual scene, and reflections and refractions are simulated by recursively casting new rays from the intersections with the geometry.
- many real-world image effects such as global illumination and other forms of complex light transport, depth of field, motion blur, etc. are more easily simulated using raytracing, and recent GPUs now feature acceleration structures to enable certain uses of raytracing in real-time graphics pipelines (e.g., NVIDIA RTX or DirectX Raytracing]).
- can be applied to both explicit and implicit representations.

**Image-based rendering**:
image-based rendering techniques generate novel images by transforming an existing set of images, typically by warping and compositing them together.

