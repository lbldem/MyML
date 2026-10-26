import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

with h5py.File('dataset_1_photons_1.hdf5', 'r') as f:
    energies = f['incident_energies'][:]
    showers = f['showers'][:]               # [:] brings f[...] pointer into our memory as a real data, as a np.array

n = len(energies)
n_train = int(n * 0.8)  # 96800

x_train = torch.tensor(energies[:n_train], dtype=torch.float32)
x_test = torch.tensor(energies[n_train:], dtype=torch.float32)
y_train = torch.tensor(showers[:n_train], dtype=torch.float32)
y_test = torch.tensor(showers[n_train:], dtype=torch.float32)

x_train_norm = torch.log(x_train)
x_test_norm = torch.log(x_test)
y_sum_train = y_train.sum(dim=1, keepdim=True)
y_train_norm = y_train / (y_sum_train + 1e-8)
y_sum_test = y_test.sum(dim=1, keepdim=True)
y_test_norm = y_test / (y_sum_test + 1e-8)

train_dataset = TensorDataset(x_train_norm, y_train_norm)
test_dataset = TensorDataset(x_test_norm, y_test_norm)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 368))

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# for epoch in range(500):
#     model.train()
#     y_pred = model(x_train_norm)
#     loss = loss_fn(y_pred, y_train_norm)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         model.eval()
#         with torch.no_grad():
#             test_loss = loss_fn(model(x_test_norm), y_test_norm)
#         print(f"epoch {epoch:3d} | train loss: {loss.item():.4e} | test loss: {test_loss.item():.4e}")

for epoch in range(100):
    # Training
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Testing
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            test_loss += loss_fn(model(x_batch), y_batch).item()
    test_loss /= len(test_loader)

    if epoch % 10 == 0:
        print(f"epoch {epoch:3d} | train loss: {train_loss:.4e} | test loss: {test_loss:.4e}")

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(true_showers.mean(axis=0).reshape(23, 16), cmap='hot', aspect='auto')
# axes[0].set_title('Geant4 (mean)')
# axes[0].set_xlabel('Cell index')
# axes[0].set_ylabel('Layer')
# axes[1].imshow(pred_showers.mean(axis=0).reshape(23,16), cmap='hot', aspect='auto')
# axes[1].set_title('Model (mean)')
# axes[1].set_xlabel('Cell index')
# energy_values = x_test.numpy().flatten()
# mask = (energy_values > 200) & (energy_values < 300)
# true_total = true_showers[mask].sum(axis=1)
# pred_total = pred_showers[mask].sum(axis=1)
# axes[2].hist(true_total, bins=50, alpha=0.5, label='Geant4', density=True, color='blue')
# axes[2].hist(pred_total, bins=50, alpha=0.5, label='Model', density=True, color='orange')
# axes[2].set_xlabel('Total energy deposit [MeV]')
# axes[2].set_ylabel('Density')
# axes[2].set_title('Total energy distribution')
# axes[2].legend()
# plt.tight_layout()
# plt.savefig('mean comparison.png')
# plt.show()

model.eval()
with torch.no_grad():
    pred_all = model(x_test_norm)
pred_showers = pred_all.numpy() * y_sum_test.numpy()
true_showers = y_test.numpy()

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
energy_list = np.unique(x_test.numpy())

for i, E in enumerate(energy_list):
    mask = x_test.numpy().flatten() == E
    true_total = true_showers[mask].sum(axis=1)
    pred_total = pred_showers[mask].sum(axis=1)

    ax = axes[i//5][i%5]
    ax.hist(true_total, bins=30, alpha=0.7, label='Geant4', density=True, color='blue')
    ax.hist(pred_total, bins=30, alpha=0.7, label='Model', density=True, color='orange')
    ax.set_title(f'E = {E/1000:.0f} ~ {2*E/1000:.0f}GeV')
    ax.set_xlabel('Total energy [MeV]')
    if i%5 == 0:
        ax.set_ylabel('Density')

axes[0][0].legend()
plt.tight_layout()
plt.savefig('energy_comparison.png')
plt.show()