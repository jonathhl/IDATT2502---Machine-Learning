import torch
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('length_weight/length_weight.csv')
y_train = train.pop('weight')
x_train = torch.tensor(train.to_numpy(), dtype=torch.double).reshape(-1, 1)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.double).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True, dtype=torch.double)
        self.b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.double)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(100000):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

# print("W = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('length')
plt.ylabel('weight')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()