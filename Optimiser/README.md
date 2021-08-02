## Gradient Descent
#### Batch Gradient Descent
A.k.a Vanilla gradient descent. 
*theta = theta - eta * Jacobian_theta(loss_over_whole_dataset)*.
```python
# psuedo code for one epoch training
jacobian = compute_gradient(loss_function, params, entire_training_dataset)
params = params - learning_rate * jacobian
```

<u>Pros</u>: it is guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces;
<u>Cons</u>: it can be very slow (huge computation cost in computing the jacobian using the entire dataset and redundant computation for similar examples in a large dataset) and intractable for datasets that don't fit in memory. Batch gradient descent is not capable of handling new examples during training either.

#### Stochastic Gradient Descent
Perform a training parameters update using ONLY 1 training example for each training step.
*theta = theta - eta * Jacobian_theta(loss_over_a_single_data)*
```python
# psuedo code for one epoch training
shuffled_training_data = random.shuffle(training_dataset)
for single_training_data in range(shuffled_training_data):
    jacobian = compute_gradient(loss_function, params, single_training_data)
    params = params - learning_rate * jacobian
```

<u>Pros</u>: Usually it is much faster than Batch Gradient Descent, and it enables the training process to jump to a new and possibly a better local minima; 
<u>Cons</u>: The loss function is always highly fluctuated, and ultimately complicates the training convergence as it sometimes overshoots.

P.S. why shuffle the dataset in each training epoch:

#### Mini-Batch Gradient Descent
A combination of Batch gradient descent and Stochastic gradient descent and it performs a training parameters update using n training examples out of N total examples for each training step.
*theta = theta - eta * Jacobian_theta(loss_over_n_data_samples)*
```python
# psuedo code for one epoch training
shuffled_training_data = random.shuffle(training_dataset)
dataset_size = sizeof(training_dataset)
batch_size = n
for i in range(dataset_size/batch_size + 1):
    batched_training_data = shuffled_training_data[i*batch_size:(i+1)*batch_size]
    jacobian = compute_gradient(loss_function, params, batched_training_data)
    params = params - learning_rate * jacobian
```

More stable convergence behaviours compared to Stochastic gradient descent and more efficient gradient computation compared to Batch gradient descent. Thus it is most used in training neural networks.

<u>Challenges to solve:</u> how to best choose learning parameters, i.e. learning rate, and how to jump out of the local minima for non-convex loss function.


## Modern Variants

#### Momentum

#### Nesterov accelerated gradient

#### Adagrad

#### Adadelta

#### RMSprop

#### Adam

#### AdaMax

#### Nadam

#### AMSGradMomentum
