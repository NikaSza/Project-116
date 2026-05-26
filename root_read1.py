import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak


file_name = "00385270_00000001_1.dvntuple.root"
file = uproot.open(file_name)
print(type(file))


keys = file.keys()
print(keys)

branches = file['MyDecayTree_muons;1'].keys()
for branch in branches:
    print(f"{branch:20s} {file['MyDecayTree_muons;1'][branch]}")


mm_tree = file["MyDecayTree_muons/DecayTree;1"]["B0_M"]
data_array = mm_tree.array(library="np")
plt.hist(data_array, bins=100, range=(1000, 4000), histtype='step', color='green')
