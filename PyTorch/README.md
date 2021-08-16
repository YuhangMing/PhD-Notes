# PyTorch-Notes

#### `torch.no_grad()`
The wrapper `with torch.no_grad()` sets all of tehe `requires_grad` flags to false.
The PyTorch official document for `autograd` is [here](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). 

In short, if a tensor has the flag `requires_grad` set to `True`, then `autograd` tracks every operation on this tensor.
"In a NN, parameters that don’t compute gradients are usually called frozen parameters. It is useful to “freeze” part of your model if you know in advance that you won’t need the gradients of those parameters (this offers some performance benefits by reducing autograd computations)."