# Random notes in CUDA

### Illegal memory access
```shell
an illegal memory access was encountered-3
```
[link](https://stackoverflow.com/questions/27277365/unspecified-launch-failure-on-memcpy) from stackoverflow.


### Synchronisation

##### Why do we need cudaDeviceSynchronize?
Although CUDA kernel launches are asynchronous, all GPUrelated tasks placed in one stream (which is the default behavior) are executed sequentially. So, for example,
```cuda
kernel1<<<X,Y>>>(...);
// kernel start execution, CPU continues to next statement
kernel2<<<X,Y>>>(...);
// kernel is placed in queue and will start after kernel1 finishes, CPU continues to next statement
cudaMemcpy(...);
// CPU blocks until memory is copied, memory copy starts only after kernel2 finishes
```
So in many cases, there is no need for `cudaDeviceSynchronize`. However, it might be useful for debugging to detect which of your kernel has caused an error (if there is any). `cudaDeviceSynchronize` may cause some slowdown, but 7-12x seems too much. Might be there is some problem with time measurement, or maybe the kernels are really fast, and the overhead of explicit synchronization is huge relative to actual
computation time.

##### `cudaDeviceSynchronize()` v.s. `__syncthreads()`
`cudaDeviceSychronize()` is used in host code (i.e. running on the CPU) when it is desired that CPU activity wait on the completion of any pending GPU activity. In many cases it's not necessary to do
this explicitly, as GPU operations issued to a single stream are automatically serialized, and certain other operations like cudaMemcpy() have an inherent blocking device synchronization built into them. But for some other purposes, such as debugging code, it may be convenient to force the device to finish any
outstanding activity.

`__syncthreads()` is used in device code (i.e. running on the GPU) and may not be necessary at all in code that has independent parallel operations (such as adding two vectors together, elementby-element). However, one example where it is commonly used is in algorithms that will operate out of shared memory. In these cases it's frequently necessary to load values from global memory into shared memory, and we want each thread in the threadblock to have an opportunity to load it's appropriate shared memory location(s), before any actual processing occurs. In this case we want to use `__syncthreads()` before the processing occurs, to ensure that shared memory is fully populated. This is just one example. `__syncthreads()` might be used any time synchronization within a block of threads is desired. It does not allow for synchronization between blocks.

The difference between `cudaMemcpy` and `cudaMemcpyAsync` is that the non-async version of the call can only be issued to stream 0 and will block the calling CPU thread until the copy is complete. The async version can optionally take a stream parameter, and returns control to the calling thread immediately, before the copy is complete. The async version typically finds usage in situations where we want to have asynchronous concurrent execution. If you have basic questions about CUDA programming, it's recommended that you take some of the webinars available.