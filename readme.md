# Machine Learning - Learning material

## 1. Linear Regression

Simple linear regression is a type of regression analysis where the number of independent variables is one and there is a linear relationship between the independent(x) and dependent(y) variable. 

A Simple Linear Equation
```
y = a_0 + a_1 * x
```

The motive of the linear regression algorithm is to find the best values for `a_0` and `a_1`.

## Cost function

The cost function helps us to figure out the best possible values for `a_0` and `a_1` which would provide the best fit line for the data points.

Since we want the best values for `a_0` and `a_1`, we convert this search problem into a minimization problem where we would like to minimize the error between the predicted value and the actual value.

![Example](https://cdn-images-1.medium.com/max/1600/1*wQCSNJ486WxL4mZ3FOYtgw.png?raw=true "Example")


## Gradient descent

 Gradient descent is a method of updating `a_0` and `a_1` to reduce the cost function(MSE). The idea is that we start with some values for `a_0` and `a_1` and then we change these values iteratively to reduce the cost. 

In the gradient descent algorithm, the number of steps you take is the learning rate. This decides on how fast the algorithm converges to the minima.

Sometimes the cost function can be a non-convex function where you could settle at a local minima but for linear regression, it is always a convex function.


![Example](https://cdn-images-1.medium.com/max/1200/1*D4Q7zeRBmZ3z1CbD37CIhg.png?raw=true "Example")


![Example](https://cdn-images-1.medium.com/max/1200/1*fr-f6K1ebanMA4Roz8JENA.png?raw=true "Example")


Alpha is the learning rate which is a hyperparameter that you must specify. A smaller learning rate could get you closer to the minima but takes more time to reach the minima, a larger learning rate converges sooner but there is a chance that you could overshoot the minima.

results :
|-------|--------------|---------------------|--------------|-------------|----------------|
|Dataset|Algorithm     |Total Dataset records|Train data set|Test data set|Accuracy        |
|titanic|classification|1309                 |418           |891          |0.791           |
|titanic|xgboost       |1309                 |418           |891          |to be calculated|
