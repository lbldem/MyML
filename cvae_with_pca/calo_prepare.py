import h5py
import numpy as np
import pickle
from sklearn.decomposition import IncrementalPCA

THRESHOLD_MEV = 0.01515
CHUNK = 5000
N_COMPONENTS = 64

def preprocess(showers, energies):
    showers = showers.copy()
    showers[showers < THRESHOLD_MEV] = 0.0  # Masking
    showers_norm = showers / energies
    return np.log1p(showers_norm * 1e4).astype(np.float32)

def iter_chunks(path):
    with h5py.File(path, 'r') as f:
        n = f['showers'].shape[0]
        for start in range(0, n, CHUNK):
            E = f['incident_energies'][start:start + CHUNK]
            S = f['showers'][start:start + CHUNK]
            yield preprocess(S, E), E.astype(np.float32)

ipca = IncrementalPCA(n_components=N_COMPONENTS)
for i, (X, _) in enumerate(iter_chunks('dataset_2_1.hdf5')):
    ipca.partial_fit(X)
    print(f"fit chunk {i+1}/20")

print(f"Explained variance: {ipca.explained_variance_ratio_.sum():.3f}")

def transform_file(path):
    Z_list, E_list = [], []
    for X, E in iter_chunks(path):
        Z_list.append(ipca.transform(X))
        E_list.append(E)
    return np.concatenate(Z_list).astype(np.float32), np.concatenate(E_list)

train_pca, train_E = transform_file('dataset_2_1.hdf5')
test_pca, test_E = transform_file('dataset_2_2.hdf5')

np.save('train_pca.npy', train_pca)
np.save('test_pca.npy', test_pca)
np.save('train_energies.npy', train_E)
np.save('test_energies.npy', test_E)

with open('pca_model.pkl', 'wb') as f:
    pickle.dump(ipca, f)

print(train_pca.shape, train_E.shape)
print("Saved.")