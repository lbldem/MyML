import numpy as np
from sklearn.metrics import roc_auc_score

R = np.load('test_voxel.npy', mmap_mode='r')
G = np.load('gen_voxel.npy', mmap_mode='r')
E = np.load('test_energies.npy')[:, 0]      # This flattens our (100000, 1) array.

def features(Y):
    f = np.expm1(Y)/1e4
    return {
        'total energy fraction': f.sum(1),
        'number of hits': (Y > 0).sum(1),
        'mean log-value of hits': Y.sum(1) / np.maximum((Y > 0).sum(1), 1),
        'max voxel fraction': f.max(1)}

for lo, hi in [(1e3, 4e3), (30e3, 60e3), (250e3, 1e6)]:
    sel = np.where((E >= lo) & (E < hi))[0][:5000]
    fr, fg = features(np.array(R[sel])), features(np.array(G[sel]))
    print(f"\n{lo/1e3:.0f} - {hi/1e3:.0f} GeV ({len(sel)} events)")
    for name in fr:
        scores = np.concatenate([fr[name], fg[name]])
        labels = np.concatenate([np.ones(len(sel)), np.zeros(len(sel))])
        auc = roc_auc_score(labels, scores)
        print(f" {name:25s} AUC = {max(auc, 1 - auc):.3f}"
              f" median-real = {np.median(fr[name]):.4g}\tmedian-gen = {np.median(fg[name]):.4g}")
