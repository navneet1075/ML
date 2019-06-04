**How do you choose which attribute to split at each node?** 

The answer is find the feature that best splits the target class into the purest possible children nodes (ie: nodes that don't contain a mix of both male and female, rather pure nodes with only one class).

This measure of purity is called the **information**. It represents the expected amount of information that would be needed to specify whether a new instance  should be classified one class or other, given the example that reached the node. 
'
**Entropy** on the other hand is a measure of impurity (the opposite). It is defined for a binary class with values a/b as.
**Entropy** = - p(a)*log(p(a)) - p(b)*log(p(b))


ends-vowel
      [9m,5f]          <--- the [..,..] notation represents the class
    /          \            distribution of instances that reached a node
   =1          =0
 -------     -------
 [3m,4f]     [6m,1f]
 
 Entropy_before = - (5/14)*log2(5/14) - (9/14)*log2(9/14) = 0.9403
 
 Entropy_left = - (3/7)*log2(3/7) - (4/7)*log2(4/7) = 0.9852
 
Entropy_right = - (6/7)*log2(6/7) - (1/7)*log2(1/7) = 0.5917

We combine the left/right entropies using the number of instances down each branch as weight factor (7 instances went left, and 7 instances went right), and get the final entropy after the split:


Entropy_after = 7/14*Entropy_left + 7/14*Entropy_right = 0.7885

**Information_Gain =** **Entropy_before - Entropy_after = 0.1518**


**At each node of the tree, this calculation is performed for every feature, and the feature with the largest information gain is chosen for the split in a greedy manner (thus favoring features that produce pure splits with low uncertainty/entropy).** This process is applied recursively from the root-node down, and stops when a leaf node contains instances all having the same class (no need to split it further).
