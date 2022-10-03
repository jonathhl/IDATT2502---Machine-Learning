import torch
import matplotlib.pyplot as plt
import numpy as np
import random

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(-1, 2)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]]).reshape(-1, 1)

class XOROperator:
    def __init__(self):
        self.W1 = torch.tensor([[random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1)]],
            requires_grad=True)
        self.b1 = torch.tensor([[random.uniform(-1, 1), random.uniform(-1, 1)]],
            requires_grad=True)
        self.W2 = torch.tensor([[random.uniform(-1, 1)], [random.uniform(-1, 1)]],
            requires_grad=True)
        self.b2 = torch.tensor([[random.uniform(-1, 1)]],
            requires_grad=True)
        
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)

model = XOROperator()

optimizer = torch.optim.SGD([model.b1, model.b2, model.W1, model.W2], 0.01)
for epoch in range(200000):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

print(f'W1 = {model.W1}, W2 = {model.W2}, b1 = {model.b1}, b2 = {model.b2}, loss = {model.loss(x_train.reshape(-1, 2), y_train)}')
