import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

TAG = 'b01'
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Conditions ────────────────────────────────────────────────────────────────────────────────────
norm = torch.load('norm_params_voxel.pt', weights_only=False)
test_E = np.load('test_energies.npy')
c = (np.log(test_E) - norm['log_E_min']) / (norm['log_E_max'] - norm['log_E_min'])

# Shuffling event numbers ───────────────────────────────────────────────────────────────────────
N = len(test_E)                                     # N = 100000
rng = np.random.default_rng(seed=0)
perm = rng.permutation(N)                           # Random permutation of 0 ~ N-1 integers
idx_train = perm[:int(0.6 * N)]                     # Slicing first 60% of the permutation
idx_val = perm[int(0.6 * N):int(0.8 * N)]           # Slicing next 20% of the permutation
idx_test = perm[int(0.8 * N):]                      # Slicing last 20% left 

# Mixed dataset (Real + Gen.) ───────────────────────────────────────────────────────────────────
NULL_TEST = False
class RealGenDataset(Dataset):
    def __init__(self, real_path, gen_path, c, idx):
        self.R = np.load(real_path, mmap_mode='r')
        self.G = np.load(gen_path, mmap_mode='r')
        self.C = torch.tensor(c, dtype=torch.float32)
        self.idx = idx
        self.partner = np.roll(idx, 1)
    def __len__(self):
        return 2 * len(self.idx)
    def __getitem__(self, k):
        j = self.idx[k // 2]
        if k % 2 == 0:
            x, cj, label = self.R[j], self.C[j], 1.0            # Even index >> real data
        elif NULL_TEST:
            jp = self.partner[k // 2]
            x, cj, label = self.R[jp], self.C[jp], 0.0
        else:
            x, cj, label = self.G[j], self.C[j], 0.0            # Odd index >> gen. data
        x = torch.from_numpy(np.array(x))
        return torch.cat([x, cj]), torch.tensor([label])

def make_loader(idx, shuffle):
    ds = RealGenDataset('test_voxel.npy', 'gen_voxel.npy', c, idx)
    return DataLoader(ds, batch_size=512, shuffle=shuffle)

train_loader = make_loader(idx_train, True)
val_loader = make_loader(idx_val, False)
test_loader = make_loader(idx_test, False)

# Classifier using DNN ───────────────────────────────────────────────────────────────────────────
clf = nn.Sequential(
    nn.Linear(6480 + 1, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 1)).to(device)               # Logit as an output
optimizer = torch.optim.Adam(clf.parameters(), lr=2e-4)

# Loss calculated if we're training / vice versa
def run_epoch(loader, train):
    clf.train() if train else clf.eval()        # Mode deviates here
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit = clf(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb, reduction='mean')
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot += loss.item() * len(xb); n += len(xb)
    return tot/n

# Training and saving best spots ────────────────────────────────────────────────────────────────
EPOCHS = 5
best_val, best_state = float('inf'), None
for epoch in range(EPOCHS):
    tr = run_epoch(train_loader, train=True)
    va = run_epoch(val_loader, train=False)
    if va < best_val:
        best_val, best_state = va, copy.deepcopy(clf.state_dict())
    print(f"Epoch {epoch+1:2d} train BCE = {tr:.4f} val BCE = {va:.4f}")

clf.load_state_dict(best_state)
torch.save(best_state, f'clf_best_{TAG}.pt')

# Scoring via AUC ───────────────────────────────────────────────────────────────────────────────
clf.eval()
probs, labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        probs.append(torch.sigmoid(clf(xb.to(device))).cpu().numpy())
        labels.append(yb.numpy())
probs = np.concatenate(probs).ravel()
labels = np.concatenate(labels).ravel()

print(f"\nTest AUC = {roc_auc_score(labels, probs):.4f}")
print(f"Test accuracy = {((probs > 0.5) == labels).mean():.4f}")

E_items = np.repeat(test_E[idx_test, 0], 2)
edges = np.logspace(np.log10(E_items.min()), np.log10(E_items.max()), 11)
bin_id = np.clip(np.digitize(E_items, edges) - 1, 0, 9)

print("\nAUC per energy bin")
for k in range(10):
    sel = bin_id == k
    auc = roc_auc_score(labels[sel], probs[sel])
    print(f"{edges[k]/1000:7.1f} - {edges[k+1]/1000:7.1f} GeV: AUC = {auc:.3f}")