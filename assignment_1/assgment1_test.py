#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 08:44:43 2017

@author: viche
"""

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

N = 20
dimensions = [10, 5, 10]
data = np.random.randn(N, dimensions[0]) 

labels = np.zeros((N, dimensions[2]))
for i in range(N):
    labels[i, random.randint(0,dimensions[2]-1)] = 1

params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )

ofs = 0
Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
ofs += Dx * H
b1 = np.reshape(params[ofs:ofs + H], (1, H))
ofs += H
W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
ofs += H * Dy
b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

z1 = data.dot(W1) + b1
hidden = sigmoid(z1) 
z2 = hidden.dot(W2) + b2
prediction = softmax(z2) 
cost = -np.sum(np.log(prediction) * labels)

# Gradient of the softmax
delta = prediction - labels

# Hidden layer
# Gradient of the W2
gradW2 = hidden.T.dot(delta)
# Gradient of the b2
gradb2 = np.sum(delta, axis = 0)


delta = delta.dot(W2.T) * sigmoid_grad(hidden)


gradW1 = data.T.dot(delta)
gradb1 = np.sum(delta, axis = 0)


grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),gradW2.flatten(), gradb2.flatten()))