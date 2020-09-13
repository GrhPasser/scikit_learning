#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Runhao G  time: 2020/9/12
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.utils.validation import check_random_state

X = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4]])
y = np.array([[3, 12, 27, 48]]).T
mlp_estimator = MLPRegressor(
                activation='logistic', # 激活函数
                # tol=1e-6,   # 连续两个iterations，损失下降未超过tol则认为收敛，当然adaptive下回将rate除以5
                # solver='lbfgs', # 对于小训练集，收敛快表现好，大训练集adam
                # shuffle=True, # 每次迭代对样本洗牌 # TODO:？？
                # learning_rate='adaptive', # 两个迭代过程，损失为下降超过tol，则将learning_rate除以5，default=0.001
                # verbose=0,
                # max_iter=1e6, # 1百万次
                # alpha=10, # L2惩罚系数
                hidden_layer_sizes=(5, 5)) # 如果要设置两层隐藏层，就用(25, 20)

hidden_layer_sizes = mlp_estimator.hidden_layer_sizes
if not hasattr(hidden_layer_sizes, "__iter__"):
    hidden_layer_sizes = [hidden_layer_sizes]
hidden_layer_sizes = list(hidden_layer_sizes)

# Validate input parameters.
# mlp_estimator._validate_hyperparameters() # 检查参数是否合理
if np.any(np.array(hidden_layer_sizes) <= 0):
    raise ValueError("hidden_layer_sizes must be > 0, got %s." %
                     hidden_layer_sizes)

# X, y = mlp_estimator._validate_input(X, y, incremental)
n_samples, n_features = X.shape

# Ensure y is 2D
# TODO:保证array为两维，即输入的y应该是np.array([[1, 2, 3]])这才是1行3列的array
# if y.ndim == 1:
#     y = y.reshape((-1, 1))

mlp_estimator.n_outputs_ = y.shape[1]

layer_units = ([n_features] + hidden_layer_sizes +
               [mlp_estimator.n_outputs_])

# check random state
mlp_estimator._random_state = check_random_state(mlp_estimator.random_state)

incremental=False
if not hasattr(mlp_estimator, 'coefs_') or (not mlp_estimator.warm_start and not
                                   incremental):
    # First time training the model
    mlp_estimator._initialize(y, layer_units)

# lbfgs does not support mini-batches
if mlp_estimator.solver == 'lbfgs':
    batch_size = n_samples
elif mlp_estimator.batch_size == 'auto':
    batch_size = min(200, n_samples)
else:
    if mlp_estimator.batch_size < 1 or mlp_estimator.batch_size > n_samples:
        warnings.warn("Got `batch_size` less than 1 or larger than "
                      "sample size. It is going to be clipped")
    batch_size = np.clip(mlp_estimator.batch_size, 1, n_samples)

# Initialize lists
activations = [X]
activations.extend(np.empty((batch_size, n_fan_out))
                   for n_fan_out in layer_units[1:])
deltas = [np.empty_like(a_layer) for a_layer in activations]

coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
              n_fan_out_ in zip(layer_units[:-1],
                                layer_units[1:])]

intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                   layer_units[1:]]

activations = mlp_estimator._forward_pass(activations)

print('output: ', activations[mlp_estimator.n_layers_ - 1])
for i in range(mlp_estimator.n_layers_ - 1):
    print('weight[%d]: ' % i, mlp_estimator.coefs_[i])
    print('bias[%d]: ' %i, mlp_estimator.intercepts_[i])
