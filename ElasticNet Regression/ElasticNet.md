# Elastic Net Regression

To minimize overfitting in Machine Learning, Reqularizations techniques are applied which helps to enhance the model's generalization performances. Elastic Net Regression is a regularization technique that combines both **L1 (Lasso)** and **L2 (Ridge)** penalties. This combination allows ElasticNet to handle scenarios where there are multiple correlated features, providing a balance between the sparsity of Lasso and the regularization of Ridge. In this article we will implement and understand the concept of Elasticnet in Sklearn.


## 🧮 Mathematical Formulation

```
minimize: 1/(2 * n_samples) * ||y - Xw||^2_2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha *(1 - l1_ratio) * ||w||^2_2
```