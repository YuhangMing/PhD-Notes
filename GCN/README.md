# Graph
Graph can be used to represent:
* social network
* web pages
* biological networks
* ...
Things that can be learnt from a graph:
* study topology and connectivity
* community detection
* identification of central nodes
* ...

## Basic Notations:
A graph ![equation](http://www.sciweavers.org/tex2img.php?eq=G%3D%28V%2C%20E%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0[/img]) is made of a set of:
* nodes, a.k.a. vertices, ![equation](http://www.sciweavers.org/tex2img.php?eq=V%3D1%2C%20...%2C%20n&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0[/img])
* edges ![equation](http://www.sciweavers.org/tex2img.php?eq=E%20%5Csubseteq%20V%20%5Ctimes%20V&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0[/img])
* An edge ![equation](http://www.sciweavers.org/tex2img.php?eq=%28i%2C%20j%29%20%5Cin%20E&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0[/img]) links nodes i and j. i an j are said to be neighbors.
* A degree of a node is its number o neighbors.

## Graph Convolutional Network (GCN)
GCN的本质：一张graph network中feature和message的流动和传播。
