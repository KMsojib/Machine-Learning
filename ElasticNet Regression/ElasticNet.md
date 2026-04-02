# Elastic Net Regression

To minimize overfitting in Machine Learning, Reqularizations techniques are applied which helps to enhance the model's generalization performances. Elastic Net Regression is a regularization technique that combines both **L1 (Lasso)** and **L2 (Ridge)** penalties. This combination allows ElasticNet to handle scenarios where there are multiple correlated features, providing a balance between the sparsity of Lasso and the regularization of Ridge. In this article we will implement and understand the concept of Elasticnet in Sklearn.


## Elastic Net Cost Function

The objective (cost) function for Elastic Net Regression is:

$$
J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 
+ \lambda_1 \sum_{j=1}^{p} |\beta_j| 
+ \lambda_2 \sum_{j=1}^{p} \beta_j^2
$$