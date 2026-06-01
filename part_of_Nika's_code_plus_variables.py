import uproot
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

root_file_path = r"/Users/barbatcs/Downloads/python_practical_2026/00385270_00000001_1.dvntuple.root"
file = uproot.open(root_file_path)

print(file.keys())

keys = file.keys()[1]
tree = file[keys]
print(tree.keys())

# reading branches from the list
all_branches = tree.keys()
variables = list(tree.keys())

### Conditions ###

def momentum_mask(pt_min=1000): 
    pt = tree["B0_PT"].array(library="np")
    return np.isfinite(pt) & (pt > pt_min)

def mcorr_mask(mcr_min=2000, mcr_max=5743):
    mcr = tree["B0_MCORR"].array(library="np")
    return np.isfinite(mcr) & (mcr > mcr_min) & (mcr < mcr_max)

def b0_ipchi2_ownpv(ipchi_max=1): 
    ipchi2 = tree["B0_IPCHI2_OWNPV"].array(library="np")
    return np.isfinite(ipchi2) & (ipchi2 < ipchi_max)

def b0_fdchi2_ownpv(fdchi2_min=500): 
    fdchi2 = tree["B0_FDCHI2_OWNPV"].array(library="np")
    return np.isfinite(fdchi2) & (fdchi2 > fdchi2_min)

def b0_dira_ownpv(b0_dira_min=0.999): 
    b0_dira = tree["B0_DIRA_OWNPV"].array(library="np")
    return np.isfinite(b0_dira) & (b0_dira > b0_dira_min)

'''def vertex_quality_mask(chi2_ndof_max=5): # < 5 is a standard, safe starting cut
    chi2 = tree["B0_ENDVERTEX_CHI2"].array(library="np")
    ndof = tree["B0_ENDVERTEX_NDOF"].array(library="np")
    
    # Calculate the reduced chi2
    reduced_chi2 = chi2 / ndof
    
    return np.isfinite(reduced_chi2) & (reduced_chi2 < chi2_ndof_max)'''

def endvertex_chi2(endvertex_chi2_max=40):
    endvertexchi2 = tree["B0_DIRA_OWNPV"].array(library="np")
    return np.isfinite(endvertexchi2) & (endvertexchi2 < endvertex_chi2_max)

def kst_892_0_pt(kst_pt_min=5565):
    kst_pt = tree["Kst_892_PT"].array(library="np")
    return np.isfinite(kst_pt) & (kst_pt > kst_pt_min)

def kplus_pt(kplus_pt_min=3523):
    kplus_pt1 = tree["Kplus_PT"].array(library="np")
    return np.isfinite(kplus_pt1) & (kplus_pt1 > kplus_pt_min)


def piminus_pt(piminus_pt_min=2067):
    piminus_pt1 = tree["piminus_PT"].array(library="np")
    return np.isfinite(piminus_pt1) & (piminus_pt1 > piminus_pt_min)


def muplus_pt(muplus_pt_min=3486):
    muplus_pt1 = tree["muplus_PT"].array(library="np")
    return np.isfinite(muplus_pt1) & (muplus_pt1 > muplus_pt_min)


def muminus_pt(muminus_pt_min=2917):
    muminus_pt1 = tree["muminus_PT"].array(library="np")
    return np.isfinite(muminus_pt1) & (muminus_pt1 > muminus_pt_min)


# IS MUON/HAS MUON:

def kplus_ismuon(k_ismu=1): 
    k_mu = tree["Kplus_isMuon"].array(library="np")
    return np.isfinite(k_mu) & (k_mu < k_ismu)

def kplus_hasmuon(k_hasmuon=1): 
    k_mu1 = tree["Kplus_hasMuon"].array(library="np")
    return np.isfinite(k_mu1) & (k_mu1 < k_hasmuon)

def piminus_ismuon(pi_ismu=1): 
    pi_mu = tree["piminus_isMuon"].array(library="np")
    return np.isfinite(pi_mu) & (pi_mu < pi_ismu)

def piminus_hasmuon(pi_hasmuon=1): 
    pi_mu1 = tree["piminus_hasMuon"].array(library="np")
    return np.isfinite(pi_mu1) & (pi_mu1 < pi_hasmuon)
                           
# PROBABILITY THAT STUFF IS KAONS:

def kaon_probability_mask(k_threshold=0.85): 
    kplus = tree["Kplus_ProbNNk"].array(library="np")
    mask = np.isfinite(kplus) & (kplus > k_threshold)
    return mask

def piminus_prob_k(pi_k_max=0.15): 
    pi_k = tree["piminus_ProbNNk"].array(library="np")
    return np.isfinite(pi_k) & (pi_k < pi_k_max)

# PROBABILITY THAT STUFF IS MUONS:

def kplus_probmu(kp_mu_max=0.15): 
    kp_mu = tree["Kplus_ProbNNmu"].array(library="np")
    return np.isfinite(kp_mu) & (kp_mu < kp_mu_max)

# PROBABILITY THAT STUFF IS ELECTRONS:

def kp_prob_e(kp_e_max=0.15): 
    kp_e = tree["Kplus_ProbNNe"].array(library="np")
    return np.isfinite(kp_e) & (kp_e < kp_e_max)

def muplus_prob_e(muplus_e_max=0.15): 
    muplus_e = tree["muplus_ProbNNe"].array(library="np")
    return np.isfinite(muplus_e) & (muplus_e < muplus_e_max)

def muminus_prob_e(muminus_e_max=0.15): 
    muminus_e = tree["muminus_ProbNNe"].array(library="np")
    return np.isfinite(muminus_e) & (muminus_e < muminus_e_max)

# PROBABILITY THAT STUFF IS PROTONS:

def kp_prob_p(kp_p_max=0.15): 
    kp_p = tree["Kplus_ProbNNp"].array(library="np")
    return np.isfinite(kp_p) & (kp_p < kp_p_max)

def muminus_prob_p(muminus_p_max=0.15): 
    muminus_p = tree["muminus_ProbNNp"].array(library="np")
    return np.isfinite(muminus_p) & (muminus_p < muminus_p_max)

def muplus_prob_p(muplus_p_max=0.15): 
    muplus_p = tree["muplus_ProbNNp"].array(library="np")
    return np.isfinite(muplus_p) & (muplus_p < muplus_p_max)

# PROBABILITY THAT STUFF IS PIONS:

def kp_prob_pi(kp_pi_max=0.15): 
    kp_pi = tree["Kplus_ProbNNpi"].array(library="np")
    return np.isfinite(kp_pi) & (kp_pi < kp_pi_max)

def muminus_prob_pi(muminus_pi_max=0.15): 
    muminus_pi = tree["muminus_ProbNNpi"].array(library="np")
    return np.isfinite(muminus_pi) & (muminus_pi < muminus_pi_max)

def muplus_prob_pi(muplus_pi_max=0.15): 
    muplus_pi = tree["muplus_ProbNNpi"].array(library="np")
    return np.isfinite(muplus_pi) & (muplus_pi < muplus_pi_max)


# combine conditions - add new masks to variable list here
def combine_masks(k_threshold=0.85, pt_min=1000, mcr_min=2000, mcr_max=5743, ipchi_max=1, kp_e_max=0.15, kp_p_max=0.15, kp_pi_max=0.15, kp_mu_max=0.15, muplus_e_max=0.15, muminus_e_max=0.15, muplus_p_max=0.15, muminus_p_max=0.15, muplus_pi_max=0.15, muminus_pi_max=0.15, pi_ismu=1, pi_hasmuon=1, k_ismu=1, k_hasmuon=1, pi_k_max=0.15, fdchi2_min=500, b0_dira_min=0.999, chi2_ndof_max=5, kst_pt_min=5565, kplus_pt_min=3523, piminus_pt_min=2067, muplus_pt_min=3486, muminus_pt_min=2917, endvertex_chi2_max=40):
    mask = (
        kaon_probability_mask(k_threshold) &
        momentum_mask(pt_min) &
        mcorr_mask(mcr_min, mcr_max) &
        #b0_ipchi2_ownpv(ipchi_max) &
        kplus_ismuon(k_ismu) &
        #kplus_hasmuon(k_hasmuon) &
        piminus_ismuon(pi_ismu) &
        #piminus_hasmuon(pi_hasmuon) &
        piminus_prob_k(pi_k_max) &
        kplus_probmu(kp_mu_max) &
        kp_prob_e(kp_e_max) &
        muminus_prob_e(muminus_e_max) &
        muplus_prob_e(muplus_e_max) &
        #kp_prob_p(kp_p_max)
        muplus_prob_p(muplus_p_max) &
        muminus_prob_p(muminus_p_max) &
        kp_prob_pi(kp_pi_max) & 
        #muplus_prob_pi(muplus_pi_max) &
        #muminus_prob_pi(muminus_pi_max)
        #b0_fdchi2_ownpv(fdchi2_min) &
        #b0_dira_ownpv(b0_dira_min) &
        #vertex_quality_mask(chi2_ndof_max) &
        #kst_892_0_pt(kst_pt_min) &
        kplus_pt(kplus_pt_min) &
        piminus_pt(piminus_pt_min) &
        muplus_pt(muplus_pt_min) &
        muminus_pt(muminus_pt_min) &
        endvertex_chi2(endvertex_chi2_max)
    )
    return mask

### Plotting two graphs next to each other ###

# FIX: Added the missing comma at the end of the first line below
def plot_with_cuts(variable, k_threshold=0.85, pt_min=1000, mcr_min=2000, mcr_max=5743, ipchi_max=1, kp_e_max=0.15, kp_p_max=0.15, kp_pi_max=0.15, kp_mu_max=0.15, muplus_e_max=0.15, muminus_e_max=0.15, muplus_p_max=0.15, muminus_p_max=0.15, muplus_pi_max=0.15, muminus_pi_max=0.15, pi_ismu=1, pi_hasmuon=1, k_ismu=1, k_hasmuon=1, pi_k_max=0.15, fdchi2_min=500, b0_dira_min=0.999, chi2_ndof_max=5, kst_pt_min=5565, kplus_pt_min=3523, piminus_pt_min=2067, muplus_pt_min= 3486, muminus_pt_min=2917, endvertex_chi2_max=40,
                   bins=100, x_min=None, x_max=None):

    data = tree[variable].array(library="np")

    # Convert y axis only if plotting B0_PT
    if variable == "B0_PT":
        data = data / 1000.0
        xlabel = "B0_PT [GeV/c]"
    else:
        xlabel = variable

    # FIX: Pass the dynamic variables to the function instead of the hardcoded default numbers
    mask = combine_masks(k_threshold=k_threshold, pt_min=pt_min, mcr_min=mcr_min, mcr_max=mcr_max, ipchi_max=ipchi_max, kp_e_max=kp_e_max, kp_p_max=kp_p_max, kp_pi_max=kp_pi_max, kp_mu_max=kp_mu_max, muplus_e_max=muplus_e_max, muminus_e_max=muminus_e_max, muplus_p_max=muplus_p_max, muminus_p_max=muminus_p_max, muplus_pi_max=muplus_pi_max, muminus_pi_max=muminus_pi_max, pi_ismu=pi_ismu, pi_hasmuon=pi_hasmuon, k_ismu=k_ismu, k_hasmuon=k_hasmuon, pi_k_max=pi_k_max, fdchi2_min=fdchi2_min, b0_dira_min=b0_dira_min, chi2_ndof_max=chi2_ndof_max, kst_pt_min=kst_pt_min, kplus_pt_min=kplus_pt_min, piminus_pt_min=piminus_pt_min, muplus_pt_min=muplus_pt_min, muminus_pt_min=muminus_pt_min, endvertex_chi2_max=endvertex_chi2_max)

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
