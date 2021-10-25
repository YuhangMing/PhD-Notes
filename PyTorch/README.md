# PyTorch-Notes

#### `torch.no_grad()`
The wrapper `with torch.no_grad()` sets all of tehe `requires_grad` flags to false.
The PyTorch official document for `autograd` is [here](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). 

In short, if a tensor has the flag `requires_grad` set to `True`, then `autograd` tracks every operation on this tensor.
"In a NN, parameters that don’t compute gradients are usually called frozen parameters. It is useful to “freeze” part of your model if you know in advance that you won’t need the gradients of those parameters (this offers some performance benefits by reducing autograd computations)."

#### `tensor.detach()`
This code creates a new tensor that shares storage with tensor that does not require grad. It remove the tensor from the computation graph. The returned tensor will never require gradient.
*Difference to `torch.no_grad()`*: 
1. `detach()` only removes one variable from the computation graph while `no_grad()` sets all the `requires_grad` flag within the `with` statement to false. 
1. `torch.no_grad()` uses less memory as it doesn't keep any intermediary results.

#### `@torch.jit.script`
In short, torch.jit can be used to enable 2x-3x speedups on custom module code by making the code execution happen in C++. Details see the blog [here](https://spell.ml/blog/pytorch-jit-YBmYuBEAACgAiv71).