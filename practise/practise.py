# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)
# x = np.linspace(0, 10, 100)
# y = 2*x + 1 + np.random.randn(100)

# plt.scatter(x, y, s=10, label='data')
# plt.xlabel('incident energy')
# plt.ylabel('detector response')
# plt.legend()
# plt.savefig('data.png')
# plt.show()

# w = 0.0
# b = 0.0
# lr = 0.005

# for epoch in range(5000):
#     y_pred = w*x + b
#     loss = np.mean((y_pred-y)**2)
#     dL_dw = np.mean(2*(y_pred-y)*x)
#     dL_db = np.mean(2*(y_pred-y))

#     w = w - lr * dL_dw
#     b = b - lr * dL_db

#     if epoch % 500 == 0:
#         print(f"epoch: {epoch:4d} | loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

# print(f"\n최종 결과 >> w: {w:.4f}, b: {b:.4f}")
# print(f"정답 >> w: 2.0000, b: 1.0000")


import torch
import torch.nn as nn
import numpy as np

# Setting up input data
np.random.seed(42)
x_np = np.linspace(0, 10, 100)
y_np = 2 * x_np + 1 + np.random.randn(100)
x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

# Defining a NN model with 1 input(x), 1 output(y = wx + b)
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Defining loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# Training loop

for epoch in range(5000):
    y_pred = model(x)               # Prediction as a NN model output
    loss = loss_fn(y_pred, y)       # Loss function using prediction and real data

    optimizer.zero_grad()           # Initialising gradients
    loss.backward()                 # Starting computation of gradients
    optimizer.step()                # Updating parameters via optimizer

    if epoch % 500 == 0:
        print(f"epoch {epoch:4d} | loss: {loss.item():.4f}")

print(f"\n최종 loss >> loss: {loss.item():.4f}")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

