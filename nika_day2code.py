import uproot
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


root_file_path = r"C:\Users\nika\OneDrive\Dokumenty\python_practical_2026\Project-116\00385292_00000001_1.dvntuple.root"  
file = uproot.open(root_file_path)

print(file.keys())

keys = file.keys()[1]   
tree = file[keys]
print(tree.keys())

#reading branches from the list
all_branches = tree.keys()
variables = list(tree.keys())

###Conditions###
#Add new functions and limits here then change only the combine_masks function to add new cuts to the final plot
#rest should be unchanged

def kaon_probability_mask(k_threshold=0.85): # change the treshold if you want more or less stict cuts
    kplus = tree["Kplus_ProbNNk"].array(library="np")
    #kminus = tree["Kmin_ProbNNk"].array(library="np")
    #kstar = tree["kst_ProbNNk"].array(library="np")

    kplus_mask = np.isfinite(kplus) & (kplus > k_threshold)
    #kminus_mask = np.isfinite(kminus) & (kminus > k_threshold)
    #kstar_mask = np.isfinite(kstar) & (kstar > k_threshold)
    return kplus_mask 

def pminus_probability_mask(pmin_threshold=0.1): # change the treshold if you want more or less stict cuts
    pminus = tree["piminus_ProbNNpi"].array(library="np")
    pminus_mask = np.isfinite(pminus) & (pminus > pmin_threshold)
    return pminus_mask

#checking if it is not a muon
def mu_isMuon_mask():
    muplus_isMuon = tree["muplus_isMuon"].array(library="np")
    muplus_mask = np.isfinite(muplus_isMuon) & (muplus_isMuon != 0)

    muminus_isMuon = tree["muminus_isMuon"].array(library="np")
    muminus_mask = np.isfinite(muminus_isMuon) & (muminus_isMuon != 0)
    return muplus_mask & muminus_mask

def b0_flight_distance_mask(min_fdchi2=100):

    flight_chi2 = tree["B0_FDCHI2_OWNPV"].array(library="np")
    flight_distance_chi2_mask = np.isfinite(flight_chi2) & (flight_chi2 > min_fdchi2)
    return (flight_distance_chi2_mask)

#combine conditions -add new masks to variable list here
def combine_masks(k_threshold=0.85, pmin_threshold=0.1, min_fdchi2=100):
    return (kaon_probability_mask(k_threshold) 
            & pminus_probability_mask(pmin_threshold) 
            & mu_isMuon_mask() 
            & b0_flight_distance_mask(min_fdchi2))
       


###Plotting two graphs next to each other
def plot_with_cuts(variable, bins=100, x_min=None, x_max=10000):

    # Load raw data (never modify this)
    data = tree[variable].array(library="np")

    # Label
    if variable == "B0_PT":
        data_plot = data / 1000.0
        xlabel = "B0_PT [GeV/c]"
        plt.yscale("log")
    else:
        data_plot = data
        xlabel = variable

    # compute data for the right graph 
    mask = combine_masks()
    filtered_data = data_plot[mask]

    # Two graphs side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].hist(data, bins=bins, range=(x_min, x_max), histtype="step")
    axes[0].set_title("Original")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Number of events")

    axes[1].hist(filtered_data, bins=bins, range=(x_min, x_max), histtype="step",color="orange")
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
pmin_threshold_slider = widgets.FloatSlider(
    value=0.1,
    min=0,
    max=1,
    step=0.01,
    description="pi- threshold:"
)
fdchi2_threshold_slider = widgets.FloatSlider(
    value=100,
    min=0,
    max=500,
    step=1,
    description="Flight Distance CHI2 threshold:"
)
widgets.interact(
    plot_with_cuts,
    variable=variable_dropdown,
    k_threshold=k_threshold_slider,
    pmin_threshold=pmin_threshold_slider,
    min_fdchi2=fdchi2_threshold_slider,
    bins=bins_slider,
    x_min=widgets.FloatText(value=None, description="x min"),
    x_max=widgets.FloatText(value=None, description="x max")
)
