This repository is a training session for the CaloChallenge, which was held in 2022 (https://arxiv.org/pdf/2410.21611).

As a novice of ML in hep-ex, I aim to learn as much as one could, on my way of trying multiple models on quite oversimplified dataset, consisting of two branches - incident energies, and shower energies.

Two branches were wrapped in hdf5 format. 

Incident energy was in shape - (100000, 1) as a np array, and shower energies were in (100000, 6480) also as a np array. 6480 comes from the voxel dissection initially defined by CaloChallenge team on GeANT4 simulation.

Up to now, I have tried VAE(Variational AutoEncoder - once with PCA and once without.), CVAE(Conditional VAE), and NFCVAE(Normalised Flow + CVAE).

Introducing different models significantly improved the expectation quality of the model.

Still got works to be done!
