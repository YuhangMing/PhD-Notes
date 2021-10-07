# Habitat-Sim
3D simulator ([link](https://github.com/facebookresearch/habitat-sim)) to operate an agent in a 3D reconstruction.

Activate ``habitat`` virtual environment.

- ScanNet Dataset:
    First convert the reconstruction from ``.ply`` file to ``.glb`` file (glTF format) using:
    ``assimp export <PLY FILE> <GLB PATH>``
    Then view it as:
    ``habitat-viewer /media/yuhang/Datasets/datasets/ScanNet/scans/scene0000_00/scene0000_00_vh_clean.glb``


- Replica Dataset:
    Run as ``habitat-viewer /media/yuhang/Datasets/datasets/Replica/office_4/habitat/mesh_semantic.ply``

Example on extracting RGB-D-Semantic images: [link](https://aihabitat.org/docs/habitat-sim/image-extractor.html#writing-custom-pose-extractors)