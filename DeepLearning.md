DeepLearning topics and bullet points


1. hyperparameters in the multi layer perceptron 
  a) number of hidden layers
  b) number of neurons per layer
  c) activation function
  d) learning rate
  e) batch size
  f) number of epochs
  
  
  Concepts and main points for reference :
  
  1. early stopping in neural networks training with params (like patience) .patience specifies how many iterations before stopping the training even if the loss does not reduce.
  2.  loss functions 
  3. activation functions : relu, tanh, sigmoid, leaky relu, softmax
  4. weight calculation : w(l) = zx(l-1)+b(l) where l is the last layer 
  5. backpropagation
  6. chain rule for calculating derivatives.
  7. benefits of different activation functions.
  8. large neural networks almost never get stuck in local minima, and even when they do these local optima are almost as good as the global optimum. However, they can still get stuck on long plateaus for a long time.
  9. trasnsfer learning : in laya man terms , when we have the lower layer already trained on some features , we can directly use those lower levels training results in directly higher hidden layers rather than training again from scratch. this is called transfer learning.
 

Batch Size :

The batch size defines the number of samples that will be propagated through the network.

For instance, let's say you have 1050 training samples and you want to set up a batch_size equal to 100. The algorithm takes the first 100 samples (from 1st to 100th) from the training dataset and trains the network. Next, it takes the second 100 samples (from 101st to 200th) and trains the network again. We can keep doing this procedure until we have propagated all samples through of the network. Problem might happen with the last set of samples. In our example, we've used 1050 which is not divisible by 100 without remainder. The simplest solution is just to get the final 50 samples and train the network.

Advantages of using a batch size < number of all samples:

It requires less memory. Since you train the network using fewer samples, the overall training procedure requires less memory. That's especially important if you are not able to fit the whole dataset in your machine's memory.

Typically networks train faster with mini-batches. That's because we update the weights after each propagation. In our example we've propagated 11 batches (10 of them had 100 samples and 1 had 50 samples) and after each of them we've updated our network's parameters. If we used all samples during propagation we would make only 1 update for the network's parameter.

Disadvantages of using a batch size < number of all samples:

The smaller the batch the less accurate the estimate of the gradient will be. In the figure below, you can see that the direction of the mini-batch gradient (green color) fluctuates much more in comparison to the direction of the full batch gradient (blue color).

Stochastic is just a mini-batch with batch_size equal to 1. In that case, the gradient changes its direction even more often than a mini-batch gradient.


* one epoch = one forward pass and one backward pass of all the training examples
* batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
* number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
