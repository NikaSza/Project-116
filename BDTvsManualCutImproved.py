# %%
import uproot 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
from scipy.integrate import quad
import ipywidgets as widgets
from IPython.display import display

# ==============================================================================
# 1. INITIALIZATION & DATA LOADING
# ==============================================================================
print("Initializing pipeline and loading ROOT file...")

root_file_path = r"/Users/barbatcs/Downloads/python_practical_2026/00385270_00000001_1.dvntuple.root"
file = uproot.open(root_file_path)
tree = file["MyDecayTree_muons/DecayTree;1"]

# Define all structural columns required across both methods
feature_names = [
    "B0_P", "B0_PT", "B0_ENDVERTEX_CHI2", "B0_IPCHI2_OWNPV", "B0_FDCHI2_OWNPV",
    "Kst_892_0_P", "Kst_892_0_PT", "Kst_892_0_ENDVERTEX_CHI2",
    "Kplus_P", "Kplus_PT", "piminus_P", "piminus_PT",
    "muplus_P", "muplus_PT", "muminus_P", "muminus_PT"
]

trigger_branches = ["B0_L0Global_TOS", "B0_Hlt1Phys_TOS"]

prob_branches = [
    "B0_MM", "Kplus_ProbNNk", "piminus_ProbNNpi", "Kplus_ProbNNpi",
    "muplus_isMuon", "muminus_isMuon", "muplus_ProbNNe", "muminus_ProbNNe",
    "muplus_ProbNNp", "muminus_ProbNNp", "muplus_ProbNNpi", "muminus_ProbNNpi",
    "eventNumber"
]

all_needed = list(set(feature_names + trigger_branches + prob_branches))

# Read everything into a single master Pandas DataFrame
df_master = tree.arrays(all_needed, library="pd")
print(f"Successfully loaded {len(df_master)} raw events.")

# ==============================================================================
# 2. MATHEMATICAL MODEL DEFINITIONS FOR FITTING
# ==============================================================================
# Composite Model for BDT (Gaussian + Exponential background)
def fit_model_composite(m, a_sig, mu, sigma, b_bkg, c_bkg):
    signal = a_sig * np.exp(-0.5 * ((m - mu) / sigma)**2)
    background = b_bkg * np.exp(-c_bkg * (m - 5000.0)) 
    return signal + background

# Separate definitions for manual fit tracking
def gaussian_signal(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

def exponential_background(x, c0, c1):
    return c0 * np.exp(c1 * (x - 5000.0))

def combined_model(x, amplitude, mean, stddev, c0, c1):
    return gaussian_signal(x, amplitude, mean, stddev) + exponential_background(x, c0, c1)

# Global variables to store dynamically extracted fit parameters for final step
fit_results = {
    'bdt': {'amp': 0, 'mu': 5280, 'sigma': 20, 'bkg_amp': 0, 'bkg_slope': 0, 'success': False},
    'manual': {'amp': 0, 'mu': 5280, 'sigma': 20, 'bkg_amp': 0, 'bkg_slope': 0, 'success': False}
}

# ==============================================================================
# 3. PIPELINE I: MANUAL CUTS & PLOTTING/FITTING
# ==============================================================================
print("\n" + "="*50)
print("EXECUTING PIPELINE I: MANUAL SELECTION")
print("="*50)

# Define Manual Cut thresholds (using your script's defaults)
k_threshold = 0.85
ipchi_max = 15.68
muplus_e_max = 0.10
muminus_e_max = 0.10
muplus_p_max = 0.10
muminus_p_max = 0.10
kp_pi_max = 0.20
muplus_pi_max = 0.70
muminus_pi_max = 0.70

manual_mask = (
    (df_master["Kplus_ProbNNk"] > k_threshold) &
    (df_master["B0_IPCHI2_OWNPV"] < ipchi_max) &
    (df_master["muminus_ProbNNe"] < muminus_e_max) &
    (df_master["muplus_ProbNNe"] < muplus_e_max) &
    (df_master["muplus_ProbNNp"] < muplus_p_max) &
    (df_master["muminus_ProbNNp"] < muminus_p_max) &
    (df_master["muplus_ProbNNpi"] < muplus_pi_max) &
    (df_master["muminus_ProbNNpi"] < muminus_pi_max) &
    (df_master["Kplus_ProbNNpi"] < kp_pi_max)
)

df_manual = df_master[manual_mask].copy()
print(f"Manual cuts applied: {len(df_manual)} events retained out of {len(df_master)}.")

# Fit Manual Mass Distribution
bins_fit = 100
mass_min, mass_max = 5000.0, 5700.0

counts_man, bin_edges_man = np.histogram(df_manual["B0_MM"], bins=bins_fit, range=(mass_min, mass_max))
bin_centers_man = (bin_edges_man[:-1] + bin_edges_man[1:]) / 2
bin_width_man = bin_edges_man[1] - bin_edges_man[0]

# Mask empty bins to ensure stable convergence
nz_man = counts_man > 0
initial_guess_man = [max(counts_man), 5280, 20, counts_man[0], 0.001]

try:
    popt_man, pcov_man = curve_fit(combined_model, bin_centers_man[nz_man], counts_man[nz_man], p0=initial_guess_man)
    fit_results['manual'] = {
        'amp': popt_man[0], 'mu': popt_man[1], 'sigma': np.abs(popt_man[2]),
        'bkg_amp': popt_man[3], 'bkg_slope': popt_man[4], 'success': True
    }
    print(f"Manual Fit Success! Mean: {popt_man[1]:.2f} MeV, Sigma: {np.abs(popt_man[2]):.2f} MeV")
except Exception as e:
    print(f"Manual selection fit failed: {e}")

# ==============================================================================
# 4. PIPELINE II: BDT TRAINING, CROSS-VALIDATION & PUNZI CUT
# ==============================================================================
print("\n" + "="*50)
print("EXECUTING PIPELINE II: BDT SELECTION")
print("="*50)

# Filter trigger requirements
trigger_cuts = (df_master["B0_L0Global_TOS"] == 1) | (df_master["B0_Hlt1Phys_TOS"] == 1)
df_bdt_pre = df_master[trigger_cuts].copy()

# Setup sideband training windows
mass_array = df_bdt_pre["B0_MM"]
signal_mask = (mass_array > 5230) & (mass_array < 5330)
background_mask = ((mass_array > 5000) & (mass_array < 5150)) | ((mass_array > 5450) & (mass_array < 5700))
training_indices = signal_mask | background_mask

y_all = np.zeros(len(df_bdt_pre))
y_all[signal_mask] = 1

even_mask = (df_bdt_pre["eventNumber"] % 2 == 0)

# Train Even/Odd Classifiers (k-fold Cross Validation)
print("Training BDT_Even on even events...")
even_train_indices = training_indices & even_mask
bdt_even = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
bdt_even.fit(df_bdt_pre[feature_names].loc[even_train_indices], y_all[even_train_indices])

print("Training BDT_Odd on odd events...")
odd_train_indices = training_indices & ~even_mask
bdt_odd = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
bdt_odd.fit(df_bdt_pre[feature_names].loc[odd_train_indices], y_all[odd_train_indices])

# Cross-folded score assignments
df_bdt_pre["BDT_Score"] = np.nan
df_bdt_pre.loc[~even_mask, "BDT_Score"] = bdt_even.predict_proba(df_bdt_pre.loc[~even_mask, feature_names])[:, 1]
df_bdt_pre.loc[even_mask, "BDT_Score"] = bdt_odd.predict_proba(df_bdt_pre.loc[even_mask, feature_names])[:, 1]

# Punzi Grid Search Scan Optimization
print("Optimizing Punzi Figure of Merit...")
cut_scan_range = np.linspace(0.0, 1.0, 200)
best_punzi_score = -1
bdt_default_cut = 0.5
a = 5.0

initial_signal_events = df_bdt_pre.loc[signal_mask]
initial_background_events = df_bdt_pre.loc[background_mask]
total_initial_signal = len(initial_signal_events)

for cut in cut_scan_range:
    passed_sig = np.sum(initial_signal_events["BDT_Score"] > cut)
    passed_bkg = np.sum(initial_background_events["BDT_Score"] > cut)
    eff_s = passed_sig / total_initial_signal if total_initial_signal > 0 else 0
    denominator = (a / 2.0) + np.sqrt(passed_bkg)
    punzi_fom = eff_s / denominator if denominator > 0 else 0
    
    if punzi_fom > best_punzi_score:
        best_punzi_score = punzi_fom
        bdt_default_cut = cut

print(f"Optimal Punzi cut value found: {bdt_default_cut:.4f} with Score: {best_punzi_score:.4f}")
df_cleaned_by_BDT = df_bdt_pre[df_bdt_pre["BDT_Score"] > bdt_default_cut].copy()

# Fit Post-BDT Mass Distribution
counts_bdt, bin_edges_bdt = np.histogram(df_cleaned_by_BDT["B0_MM"], bins=bins_fit, range=(mass_min, mass_max))
bin_centers_bdt = (bin_edges_bdt[:-1] + bin_edges_bdt[1:]) / 2

initial_guesses_bdt = [max(counts_bdt), 5280, 20, min(counts_bdt), 0.001]
try:
    popt_bdt, _ = curve_fit(fit_model_composite, bin_centers_bdt, counts_bdt, p0=initial_guesses_bdt)
    fit_results['bdt'] = {
        'amp': popt_bdt[0], 'mu': popt_bdt[1], 'sigma': np.abs(popt_bdt[2]),
        'bkg_amp': popt_bdt[3], 'bkg_slope': popt_bdt[4], 'success': True
    }
    print(f"BDT Fit Success! Mean: {popt_bdt[1]:.2f} MeV, Sigma: {np.abs(popt_bdt[2]):.2f} MeV")
except Exception as e:
    print(f"BDT selection fit failed: {e}")


# ==============================================================================
# 5. PIPELINE III: RE-DEFINING SIGNAL FUNCTIONS WITH DYNAMIC RESULTS & COMPARING
# ==============================================================================
print("\n" + "="*50)
print("EXECUTING PIPELINE III: MATHEMATICAL AREA INTEGRATION COMPARISON")
print("="*50)

if not fit_results['bdt']['success'] or not fit_results['manual']['success']:
    print("Warning: One of your models failed to fit properly. Using placeholders fallback values for integration demo.")
    p_bdt = [6554.16, 5279.74, 18.11, 0.0, 0.0]
    p_man = [5607.87, 5280.07, 17.26, 511.71, -0.00391]
else:
    p_bdt = [fit_results['bdt']['amp'], fit_results['bdt']['mu'], fit_results['bdt']['sigma'], fit_results['bdt']['bkg_amp'], fit_results['bdt']['bkg_slope']]
    p_man = [fit_results['manual']['amp'], fit_results['manual']['mu'], fit_results['manual']['sigma'], fit_results['manual']['bkg_amp'], fit_results['manual']['bkg_slope']]

# Redefining standard numerical math operations dynamically using parameter mappings
def dynamic_bdt_signal(mass):
    return p_bdt[0] * np.exp(-0.5 * ((mass - p_bdt[1]) / p_bdt[2])**2)

def dynamic_bdt_background(mass):
    return p_bdt[3] * np.exp(-p_bdt[4] * (mass - 5000.0))

def dynamic_bdt_total(mass):
    return dynamic_bdt_signal(mass) + dynamic_bdt_background(mass)

def dynamic_manual_signal(mass):
    return p_man[0] * np.exp(-0.5 * ((mass - p_man[1]) / p_man[2])**2)

def dynamic_manual_background(mass):
    slope = p_man[4] if p_man[4] < 0 else -p_man[4]
    return p_man[3] * np.exp(slope * (mass - 5000.0))

def dynamic_manual_total(mass):
    return dynamic_manual_signal(mass) + dynamic_manual_background(mass)

# FIX: Calculate bin width to correct the integral values
bin_width = (mass_max - mass_min) / bins_fit

# Numerical Definite Integration using Quadpack (Divided by bin_width to get true event counts)
bdt_sig_area, _ = quad(dynamic_bdt_signal, mass_min, mass_max)
bdt_sig_events = bdt_sig_area / bin_width

bdt_bkg_area, _ = quad(dynamic_bdt_background, mass_min, mass_max)
bdt_bkg_events = bdt_bkg_area / bin_width

bdt_total_events = bdt_sig_events + bdt_bkg_events
bdt_purity = (bdt_sig_events / bdt_total_events) * 100 if bdt_total_events > 0 else 0

manual_sig_area, _ = quad(dynamic_manual_signal, mass_min, mass_max)
manual_sig_events = manual_sig_area / bin_width

manual_bkg_area, _ = quad(dynamic_manual_background, mass_min, mass_max)
manual_bkg_events = manual_bkg_area / bin_width

manual_total_events = manual_sig_events + manual_bkg_events
manual_purity = (manual_sig_events / manual_total_events) * 100 if manual_total_events > 0 else 0

# Comparative Analysis Area Plot
m_axis = np.linspace(mass_min, mass_max, 1000)
plt.figure(figsize=(12, 7))

# --- NEW: Plot the actual BDT Data Points ---
plt.errorbar(bin_centers_bdt, counts_bdt, yerr=np.sqrt(counts_bdt), fmt='o', color='darkgreen', 
             markersize=5, alpha=0.7, label='BDT Data')

# --- NEW: Plot the actual Manual Data Points ---
plt.errorbar(bin_centers_man, counts_man, yerr=np.sqrt(counts_man), fmt='s', color='darkred', 
             markersize=5, alpha=0.7, label='Manual Data')

# Plot the fits
plt.plot(m_axis, dynamic_bdt_total(m_axis), color='limegreen', linestyle='-', linewidth=2.5,
         label=rf'BDT Fit ($\mu$={p_bdt[1]:.2f}, $\sigma$={p_bdt[2]:.2f})')

plt.plot(m_axis, dynamic_manual_total(m_axis), color='salmon', linestyle='-', linewidth=2.5,
         label=rf'Manual Fit ($\mu$={p_man[1]:.2f}, $\sigma$={p_man[2]:.2f})')

# Shade background contamination
plt.plot(m_axis, dynamic_manual_background(m_axis), color='red', linestyle='--', linewidth=1.5)
plt.fill_between(m_axis, 0, dynamic_manual_background(m_axis), color='red', alpha=0.1, label='Manual Background')

if p_bdt[3] > 0.1: 
    plt.plot(m_axis, dynamic_bdt_background(m_axis), color='green', linestyle=':', linewidth=1.5)
    plt.fill_between(m_axis, 0, dynamic_bdt_background(m_axis), color='green', alpha=0.1, label='BDT Background')

plt.title(r'Analytical & Area Comparison: BDT vs. Manual Cuts ($B^0 \rightarrow K^{*0}\mu^+\mu^-$)', fontsize=14)
plt.xlabel('Invariant Mass $m$ [MeV/$c^2$]', fontsize=12)
plt.ylabel(f'Events / {bin_width:.1f} MeV/$c^2$', fontsize=12)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Final Quantitative Performance Metrics Output Report
print("="*75)
print(f"{'METRIC':<28} | {'BDT CUT METHOD':<20} | {'MANUAL CUT METHOD':<20}")
print("="*75)
print(f"{'Mass Mean (mu)':<28} | {f'{p_bdt[1]:.2f} MeV':<20} | {f'{p_man[1]:.2f} MeV':<20}")
print(f"{'Mass Resolution (sigma)':<28} | {f'{p_bdt[2]:.2f} MeV':<20} | {f'{p_man[2]:.2f} MeV':<20}")
print("-"*75)
print(f"{'SIGNAL EVENTS':<28} | {bdt_sig_events:<20.0f} | {manual_sig_events:<20.0f}")
print(f"{'BACKGROUND EVENTS':<28} | {bdt_bkg_events:<20.0f} | {manual_bkg_events:<20.0f}")
print(f"{'TOTAL EVENTS under fit':<28} | {bdt_total_events:<20.0f} | {manual_total_events:<20.0f}")
print("-"*75)
print(f"{'SAMPLE PURITY (%)':<28} | {bdt_purity:<19.2f}% | {manual_purity:<19.2f}%")
print("="*75)
