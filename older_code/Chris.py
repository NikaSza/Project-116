import uproot
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display



root_file_path = r"C:\Users\chris\Downloads\00385270_00000001_1.dvntuple.root" 
file = uproot.open(root_file_path)

print(file.keys())

keys = file.keys()[1]   
tree = file[keys]

print(tree.keys())

def plot(variable, bins=100, x_min=None, x_max=None):
    data = tree[variable].array(library="np")

    # Remove invalid values
    data = data[np.isfinite(data)]

    if variable == "B0_PT":
        data = data / 1000.0 

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, range=(x_min, x_max), histtype="step")
    plt.show()

variables = list(tree.keys())

pt = tree["B0_PT"].array(library="np") / 1000.0
kplus = tree["Kplus_ProbNNk"].array(library="np")

mask_all = np.isfinite(pt) & np.isfinite(kplus)
np.isfinite(kplus)
musK_cut = mask_all & (kplus > 0.85)

plt.figure(figsize=(8, 5))

plt.hist(
    pt[mask_all],
    bins=100,
    range=(0, 100),
    histtype="step",
    linewidth=2,
    label = "All"

)
plt.hist(
    pt[musK_cut],
    bins=100,
    range=(0, 100),
    histtype="step",
    linewidth=2,
    label = "kplus > 0.85"
)
plt.yscale("log")
plt.xlabel(r"$B^0$ $p_T$ [GeV/c]")
plt.ylabel("Counts")
plt.title(r"$B^0$ $p_T$ Distribution before and after kplus Cut")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
variable_dropdown = widgets.Dropdown(
    options=variables,
    description="Variable:"
)

bins_slider = widgets.IntSlider(
    value=100,
    min=1,
    max=300,
    step=1,
    description="Bins:"
)

widgets.interact(plot, variable=variable_dropdown, bins=bins_slider, x_min=widgets.FloatText(value=None, description="x min"), x_max=widgets.FloatText(value=None, description="x max")
)


print(tree.keys())
