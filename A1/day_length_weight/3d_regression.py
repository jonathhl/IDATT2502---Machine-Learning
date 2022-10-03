import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_csv('day_length_weight/day_length_weight.csv', dtype='float')
y_train = train.pop('length')
x_train = torch.tensor(train.to_numpy(), dtype=torch.float).reshape(-1, 2)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float).reshape(-1, 1)

class LinearRegressionModel:
    def __init__(self):
        self.W = torch.rand((2,1), requires_grad=True)
        self.b = torch.rand((1,1), requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        x = torch.nn.functional.mse_loss(self.f(x), y)
        return x

model = LinearRegressionModel()

optimizer = torch.optim.Adam([model.b, model.W], lr=0.0001)
for epoch in range(100000):
    model.loss(x_train, y_train).backward() 
    optimizer.step() 
    optimizer.zero_grad()  

print("W = %s, b = %s, loss = %s" %(model.W, model.b, model.loss(x_train, y_train)))

xt =x_train.t()[0]
yt =x_train.t()[1]

fig = plt.figure('Linear regression 3d')
ax = fig.add_subplot(projection='3d')
# Plot
ax.scatter(xt.numpy(),  yt.numpy(), y_train.numpy(),label='$(x^{(i)},y^{(i)}, z^{(i)})$')
ax.scatter(xt.numpy(),yt.numpy() ,model.f(x_train).detach().numpy() , label='$\\hat y = f(x) = xW+b$', c="orange")
ax.set_xlabel('Days')
ax.set_ylabel('Weight')
ax.set_zlabel('Length')
ax.legend()
plt.show()