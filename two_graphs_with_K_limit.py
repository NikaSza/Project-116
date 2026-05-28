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
variables = list(tree.keys())

###Conditions###
#Add new functions and limits here 


def kaon_probability_mask(k_threshold=0.85): # change the treshold if you want more or less stict cuts
    kplus = tree["Kplus_ProbNNk"].array(library="np")
    kplus_mask = np.isfinite(kplus) & (kplus > k_threshold)
    return kplus_mask

#example of condition to be changed later/ add similar functions below
def pminus_probability_mask(pmin_threshold=0.1): # change the treshold if you want more or less stict cuts
    pminus = tree["piminus_ProbNNpi"].array(library="np")
    pminus_mask = np.isfinite(pminus) & (pminus > pmin_threshold)
    return pminus_mask

#combine conditions -add new masks to variable list here
def combine_masks(k_threshold=0.85, pmin_threshold=0.1):
    return (
        kaon_probability_mask(k_threshold)
        & pminus_probability_mask(pmin_threshold)
    )


###Plotting two graphs next to each other

def plot_with_cuts(variable, k_threshold=0.85, pmin_threshold=0.1, bins=100, x_min=None, x_max=10000):

    # Load raw data (never modify this)
    data = tree[variable].array(library="np")

    # Label
    if variable == "B0_PT":
        data_plot = data / 1000.0
        xlabel = "B0_PT [GeV/c]"
    else:
        data_plot = data
        xlabel = variable

    # compute data for the right graph 
    mask = combine_masks(k_threshold, pmin_threshold)
    filtered_data = data_plot[mask]

    # Two graphs side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].hist(data, bins=bins, range=(x_min, x_max), histtype="step")
    axes[0].set_title("Original")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Number of events")

    axes[1].hist(filtered_data, bins=bins, range=(x_min, x_max), histtype="step")
    axes[1].set_title(f"After cuts: ")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Number of events")

    #plt.yscale("log")
    plt.tight_layout()
    plt.show()

    print("Original events:", len(data))
    print("Events after cut:", len(filtered_data))
    print("Removed events:", len(data) - len(filtered_data))


###Interactive part###
#choose variable to plot
variable_dropdown = widgets.Dropdown(
    options=variables,
    description="Variable:"
)
#sliders for cuts
bins_slider = widgets.IntSlider(
    value=100,
    min=1,
    max=300,
    step=1,
    description="Bins:"
)
k_threshold_slider = widgets.FloatSlider(
    value=0.85,
    min=0,
    max=1,
    step=0.01,
    description="K+ threshold:"
)
pm_threshold_slider = widgets.FloatSlider(
    value=0.1,
    min=0,
    max=1,
    step=0.01,
    description="pi- threshold:"
)
widgets.interact(
    plot_with_cuts,
    variable=variable_dropdown,
    k_threshold=k_threshold_slider,
    pmin_threshold=pm_threshold_slider,
    bins=bins_slider,
    x_min=widgets.FloatText(value=None, description="x min"),
    x_max=widgets.FloatText(value=None, description="x max")
)
