overfitting in the decision trees :

random noise 
incorrect labelling


solutions :

1. dont allow the tree to grow beyond  a certain point where it perfectly classifies the training data
2. allow the tree to overfit the data (means dont stop growth at all) and then post prune the tree.


Although the first of these approaches might seem.more direct, the second
approach of post-pruning overfit trees has been found to be more successful in
practice. This is due to the difficulty in the first approach of estimating precisely
when to stop growing the tree.


* Regardless of whether the correct tree size is found by stopping early or
by post-pruning, a key question is what criterion is to be used to determine the
correct final tree size. 

Approaches include:
1. Use a separate set of examples, distinct from the training examples, to evaluate
the utility of post-pruning nodes from the tree.

2.Use all the available data for training, but apply a statistical test to estimate
whether expanding (or pruning) a particular node is likely to produce an
improvement beyond the training set. For example, Quinlan (1986) uses a
chi-square test to estimate whether further expanding a node is likely to
improve performance over the entire instance distribution, or only on the
current sample of training data.

3. Use an explicit measure of the complexity for encoding the training examples
and the decision tree, halting growth of the tree when this encoding
size is minimized. This approach, based on a heuristic is called the Minimum
Description Length principle.
