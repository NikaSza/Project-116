import uproot
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display



root_file_path = r"C:\Users\nika\OneDrive\Dokumenty\python_practical_2026\Project-116\00385270_00000001_1.dvntuple.root"  
file = uproot.open(root_file_path)

print(file.keys())
keys = file.keys()[0]  


def plot(variable, bins=100, x_min=None, x_max=None):
    data = tree[variable].array(library="np")

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, range=(x_min, x_max), histtype="step")
    plt.show()

variables = list(tree.keys())

variable_dropdown = widgets.Dropdown(
    options=variables,
    description="Variable:"
)
bins_slider = widgets.IntSlider(
    value=100,
    min=10,
    max=300,
    step=10,
    description="Bins:"
)
widgets.interact(plot, variable=variable_dropdown, bins=bins_slider, x_min=widgets.FloatText(value=None, description="x min"), x_max=widgets.FloatText(value=None, description="x max")
)
