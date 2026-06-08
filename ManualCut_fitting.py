#CURVE FITTING



import uproot

import numpy as np

import matplotlib.pyplot as plt

import ipywidgets as widgets

from IPython.display import display

from scipy.optimize import curve_fit





root_file_path = r"C:\Users\Jonas\Documents\Maastricht University\MSP\Year 1\Period 6\Root_files\00385270_00000001_1.dvntuple.root"

file = uproot.open(root_file_path)



print(file.keys())



keys = file.keys()[1]

tree = file[keys]

print(tree.keys())



# reading branches from the list

all_branches = tree.keys()

variables = list(tree.keys())



### Conditions ###





def b0_ipchi2_ownpv(ipchi_max=15.68):

    ipchi2 = tree["B0_IPCHI2_OWNPV"].array(library="np")

    return np.isfinite(ipchi2) & (ipchi2 < ipchi_max)





                           

# PROBABILITY THAT STUFF IS KAONS:



def kaon_probability_mask(k_threshold=0.85):

    kplus = tree["Kplus_ProbNNk"].array(library="np")

    mask = np.isfinite(kplus) & (kplus > k_threshold)

    return mask





# PROBABILITY THAT STUFF IS ELECTRONS:





def muplus_prob_e(muplus_e_max=0.10):

    muplus_e = tree["muplus_ProbNNe"].array(library="np")

    return np.isfinite(muplus_e) & (muplus_e < muplus_e_max)



def muminus_prob_e(muminus_e_max=0.10):

    muminus_e = tree["muminus_ProbNNe"].array(library="np")

    return np.isfinite(muminus_e) & (muminus_e < muminus_e_max)



# PROBABILITY THAT STUFF IS PROTONS:



def muminus_prob_p(muminus_p_max=0.10):

    muminus_p = tree["muminus_ProbNNp"].array(library="np")

    return np.isfinite(muminus_p) & (muminus_p < muminus_p_max)



def muplus_prob_p(muplus_p_max=0.10):

    muplus_p = tree["muplus_ProbNNp"].array(library="np")

    return np.isfinite(muplus_p) & (muplus_p < muplus_p_max)



# PROBABILITY THAT STUFF IS PIONS:



def kp_prob_pi(kp_pi_max=0.20):

    kp_pi = tree["Kplus_ProbNNpi"].array(library="np")

    return np.isfinite(kp_pi) & (kp_pi < kp_pi_max)



def muminus_prob_pi(muminus_pi_max=0.70):

    muminus_pi = tree["muminus_ProbNNpi"].array(library="np")

    return np.isfinite(muminus_pi) & (muminus_pi < muminus_pi_max)



def muplus_prob_pi(muplus_pi_max=0.70):

    muplus_pi = tree["muplus_ProbNNpi"].array(library="np")

    return np.isfinite(muplus_pi) & (muplus_pi < muplus_pi_max)





# combine conditions - add new masks to variable list here

def combine_masks(k_threshold=0.85, ipchi_max=1, kp_pi_max=0.15, muplus_e_max=0.15, muminus_e_max=0.15, muplus_p_max=0.15, muminus_p_max=0.15, muplus_pi_max=0.15, muminus_pi_max=0.15):

    mask = (

        kaon_probability_mask(k_threshold) &

        b0_ipchi2_ownpv(ipchi_max) &

        muminus_prob_e(muminus_e_max) &

        muplus_prob_e(muplus_e_max) &

        muplus_prob_p(muplus_p_max) &

        muminus_prob_p(muminus_p_max) &

        muplus_prob_pi(muplus_pi_max) &

        muminus_prob_pi(muminus_pi_max) &

        kp_prob_pi(kp_pi_max)    

    )

    return mask



### Plotting two graphs next to each other ###



def plot_with_cuts(variable, k_threshold=0.85, ipchi_max=15.68, kp_pi_max=0.20, muplus_e_max=0.10, muminus_e_max=0.10, muplus_p_max=0.10, muminus_p_max=0.10, muplus_pi_max=0.70, muminus_pi_max=0.70,

                   bins=100, x_min=None, x_max=None):



    data = tree[variable].array(library="np")



    # Convert y axis only if plotting B0_PT

    if variable == "B0_PT":

        data = data / 1000.0

        xlabel = "B0_PT [GeV/c]"

    else:

        xlabel = variable



    # FIX: Pass the dynamic variables to the function instead of the hardcoded default numbers

    mask = combine_masks(k_threshold=k_threshold, kp_pi_max=kp_pi_max, ipchi_max=ipchi_max, muplus_e_max=muplus_e_max, muminus_e_max=muminus_e_max, muplus_p_max=muplus_p_max, muminus_p_max=muminus_p_max, muplus_pi_max=muplus_pi_max, muminus_pi_max=muminus_pi_max)



    filtered_data = data[mask]



    ### Two graphs side by side ###

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



    # --- NEW: Run the fit dynamically if we are looking at the mass! ---

    if variable == "B0_MM":

        print("\n--- Running Mass Fit ---")

       

        # Use the widget's x_min and x_max if you typed them in,

        # otherwise default to the safe B0 fit window (5100 - 5400)

        fit_min = x_min if x_min is not None else 5100

        fit_max = x_max if x_max is not None else 5400

       

        # Count how many events are actually inside our fit window

        events_in_fit_window = np.sum((filtered_data >= fit_min) & (filtered_data <= fit_max))

        print(f"Total events in cut: {len(filtered_data)}")

        print(f"Events inside fit window ({fit_min}-{fit_max}): {events_in_fit_window}")

       

        # Run the fit!

        fit_and_plot_mass(filtered_data, bins=bins, mass_min=fit_min, mass_max=fit_max)







### Interactive part ###



variable_dropdown = widgets.Dropdown(

    options=variables,

    value="B0_MM",

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

    x_max=widgets.FloatText(value=9000, description="x max")



)





# 1. Define the mathematical models (Updated with m - 5000 shift)
def gaussian_signal(x, amplitude, mean, stddev):
    """The signal peak"""
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

def exponential_background(x, c0, c1):
    """The combinatorial background shifted to avoid underflow/overflow"""
    return c0 * np.exp(c1 * (x - 5000.0))

def combined_model(x, amplitude, mean, stddev, c0, c1):
    """Signal + Background"""
    return gaussian_signal(x, amplitude, mean, stddev) + exponential_background(x, c0, c1)

def fit_and_plot_mass(mass_data, bins=300, mass_min=4000, mass_max=6000):
    # 2. Create the histogram data to fit
    counts, bin_edges = np.histogram(mass_data, bins=bins, range=(mass_min, mass_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    nonzero_mask = counts > 0
    x_data = bin_centers[nonzero_mask]
    y_data = counts[nonzero_mask]

    # 2. Create the histogram data to fit

    # We need the x (bin centers) and y (number of events per bin) points

    counts, bin_edges = np.histogram(mass_data, bins=bins, range=(mass_min, mass_max))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_width = bin_edges[1] - bin_edges[0]

   

    # Ignore empty bins to avoid division-by-zero errors in advanced fitting

    nonzero_mask = counts > 0

    x_data = bin_centers[nonzero_mask]

    y_data = counts[nonzero_mask]



    # 3. Provide initial guesses for the fit parameters
    # [amplitude, mean, stddev, background_c0, background_c1]
    # FIX: counts[0] corresponds to roughly mass=5000, which aligns perfectly with our shift!
    initial_guess = [max(counts), 5279, 20, counts[0], -0.001]

   
# 4. Perform the curve fit
    try:
        popt, pcov = curve_fit(combined_model, x_data, y_data, p0=initial_guess)
        amplitude, mean, stddev, c0, c1 = popt
        
        # 5. Calculate the Number of Signal Events
        n_signal_events = (amplitude * np.abs(stddev) * np.sqrt(2 * np.pi)) / bin_width
        
        # Calculate errors on the parameters
        perr = np.sqrt(np.diag(pcov))
        n_signal_error = (perr[0] * np.abs(stddev) * np.sqrt(2 * np.pi)) / bin_width 
        
        print(f"--- Fit Results ---")
        print(f"B0 Mass Mean: {mean:.2f} MeV/c^2")
        print(f"Mass Resolution (StdDev): {np.abs(stddev):.2f} MeV/c^2")
        print(f"Number of Signal Events: {n_signal_events:.0f} ± {n_signal_error:.0f}")
        
        # Print the Analytical Extracted Function for Manual Cuts
        print("\n" + "="*50)
        print("EXTRACTED ANALYTICAL FUNCTION (MANUAL CUTS)")
        print("="*50)
        # FIX: Explicitly printing the shifted function format
        print(f"Post-manual-cut Function f(m):\n  {amplitude:.2f} * exp(-0.5*((m - {mean:.2f})/{np.abs(stddev):.2f})^2) + {c0:.2f} * exp({c1:.5f} * (m - 5000))")
        print("="*50 + "\n")
        
    except RuntimeError:
        print("Error: Curve fit failed to converge. Try adjusting the initial guesses or cuts.")
        return

       



   # 6. Plotting the results
    x_plot = np.linspace(mass_min, mass_max, 500)
    
    plt.figure(figsize=(8, 6))
    
    # Plot original data as points with error bars (Poisson errors)
    plt.errorbar(x_data, y_data, yerr=np.sqrt(y_data), fmt='o', color='black', label='Data', markersize=4)
    
    # Plot the total fit
    plt.plot(x_plot, combined_model(x_plot, *popt), color='red', linewidth=2, label='Total Fit')
    
    # Plot the signal component
    plt.plot(x_plot, gaussian_signal(x_plot, amplitude, mean, stddev), color='blue', linestyle='--', label='Signal (Gaussian)')
    
    # Plot the background component
    plt.plot(x_plot, exponential_background(x_plot, c0, c1), color='green', linestyle='--', label='Background (Exponential)')
    
    plt.title("B0 Mass Distribution Fit")
    plt.xlabel("B0_MM [MeV/c^2]")
    plt.ylabel(f"Events / {bin_width:.1f} MeV/c^2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
