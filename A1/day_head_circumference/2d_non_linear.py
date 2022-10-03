import matplotlib.pyplot as plt
import pandas as pd
import torch

data = pd.read_csv('day_head_circumference/day_head_circumference.csv')
y_train = data.pop('head circumference')
x_train = torch.tensor(data.to_numpy(), dtype=torch.float)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float).reshape(-1, 1)

class NonLinearRegressionModel:
    
    def __init__(self, max):
        self.max = max
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = NonLinearRegressionModel(data.shape[0])

optimizer = torch.optim.SGD([model.W, model.b], 0.0000001)
for epoch in range (200000):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

# print("W = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

plt.figure('Nonlinear regression 2d')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_train, y_train)
x = torch.arange(torch.min(x_train), torch.max(x_train), 1.0).reshape(-1, 1)
y = model.f(x).detach()
plt.plot(x, y, color='orange',
         label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$')

plt.legend()
plt.show()