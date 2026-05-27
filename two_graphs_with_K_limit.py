import uproot
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


root_file_path = r"C:\Users\nika\OneDrive\Dokumenty\python_practical_2026\Project-116\00385270_00000001_1.dvntuple.root"  
file = uproot.open(root_file_path)

print(file.keys())

keys = file.keys()[1]   
tree = file[keys]
print(tree.keys())


#reading branches from the list
all_branches = tree.keys()

#setting up axis
px_candidates = [b for b in all_branches if "px" in b.lower()]
py_candidates = [b for b in all_branches if "py" in b.lower()]


px_branch_auto = px_candidates[0]
py_branch_auto = py_candidates[0]


###Conditions###
#Add new functions and limits here 
variables = list(tree.keys())

def kaon_probability_mask(k_threshold=0.85): # change the treshold if you want more or less stict cuts
    kplus = tree["Kplus_ProbNNk"].array(library="np")

    mask = np.isfinite(kplus) & (kplus > k_threshold)

    return mask

###Plotting two graphs next to each other

def plot_with_cuts(variable, k_threshold=0.85,
                             bins=100, x_min=None, x_max=None):

    data = tree[variable].array(library="np")

    # Convert y axis only if plotting B0_PT 
    if variable == "B0_PT":
        data = data / 1000.0
        xlabel = "B0_PT [GeV/c]"
    else:
        xlabel = variable

    mask = kaon_probability_mask(k_threshold)

    filtered_data = data[mask]


###Two graphs side by side

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].hist(data, bins=bins, range=(x_min, x_max), histtype="step")
    axes[0].set_title("Original")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Number of events")

    axes[1].hist(filtered_data, bins=bins, range=(x_min, x_max), histtype="step")
    axes[1].set_title(f"After cuts: Kplus_ProbNNk > {k_threshold}")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Number of events")

    plt.tight_layout()
    plt.show()

    print("Original events:", len(data))
    print("Events after cut:", len(filtered_data))
    print("Removed events:", len(data) - len(filtered_data))

###Interactive part###

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


widgets.interact(
    plot_with_cuts,
    variable=variable_dropdown,
    bins=bins_slider,
    x_min=widgets.FloatText(value=None, description="x min"),
    x_max=widgets.FloatText(value=None, description="x max")
)
