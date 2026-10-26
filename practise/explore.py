import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('dataset_1_photons_1.hdf5', 'r') as f:
    print("==== Uppermost structure ====")
    for key in f.keys():
        print(key, f[key].shape, f[key].dtype)
    energies = f['incident_energies'][:]    ## 121000 events, single incident energy
    showers = f['showers'][:]               ## 121000 events, energy deposits in 368 cells

print("\n==== Incident energies ====")
print("min: ", energies.min())
print("max: ", energies.max())
print("sample: ", energies[:5])

print("\n==== Shower data ====")
print("min: ", showers.min())
print("max: ", showers.max())
print("sample (first event): ", showers[0,:5])  ## [x,:y] this is a double slicing. up to y-th cell of x-th event

# shower = showers[0].reshape(23, 16)
# plt.figure(figsize=(6,8))
# plt.imshow(shower, cmap='hot', aspect='auto')
# plt.colorbar(label='Energy deposit [MeV]')
# plt.xlabel('Cell index')
# plt.ylabel('Layer')
# plt.title(f'Shower (incident energy: {energies[0][0]:.0f} MeV)')
# plt.savefig('shower.png')
# print("\nSaving done.")

mean_shower = showers.mean(axis=0).reshape(23, 16)
plt.figure(figsize=(6,8))
plt.imshow(mean_shower, cmap='hot', aspect='auto')
plt.colorbar(label='Mean energy deposit [MeV]')
plt.xlabel('Cell index')
plt.ylabel('Layer')
plt.title('Average shower (121000 events)')
plt.savefig('Mean_shower.png')
plt.show()