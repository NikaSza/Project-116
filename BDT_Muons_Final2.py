# %%
import uproot 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# We open the ROOT file and access the tree
file = uproot.open(r"C:\Users\Jonas\Documents\Maastricht University\MSP\Year 1\Period 6\Root_files\00385270_00000001_1.dvntuple.root")
tree = file["MyDecayTree_muons/DecayTree;1"]

# We define the feature names for our BDT training
feature_names = [
    "B0_P", "B0_PT", "B0_ENDVERTEX_CHI2", "B0_IPCHI2_OWNPV", "B0_FDCHI2_OWNPV",
    "Kst_892_0_P", "Kst_892_0_PT", "Kst_892_0_ENDVERTEX_CHI2",
    "Kplus_P", "Kplus_PT", "piminus_P", "piminus_PT",
    "muplus_P", "muplus_PT", "muminus_P", "muminus_PT"
]

# We add the Trigger branches to the raw data we will use for training
trigger_branches = ["B0_L0Global_TOS", "B0_Hlt1Phys_TOS"]

# We prepare the DataFrame for training by selecting the relevant features and the target variable
all_needed = feature_names + ["B0_MM", "Kplus_ProbNNk", "piminus_ProbNNpi", "muplus_isMuon", "muminus_isMuon", "eventNumber"] + trigger_branches
df_raw = tree.arrays(all_needed, library="pd")

# We filter the data based on the following conditions: (we removed the particle ID cuts and the kinematic cuts)
cut_conditions = (
    # Trigger cuts
    ((df_raw["B0_L0Global_TOS"] == 1) | (df_raw["B0_Hlt1Phys_TOS"] == 1))
)

# We clean the data by applying the cut conditions
df_clean = df_raw[cut_conditions].copy()

# We create a boolean mask to separate even and odd events based on the eventNumber
even_mask = (df_clean["eventNumber"] % 2 == 0)

# We define the BDT Mass Windows on our pre-selected data
mass_array = df_clean["B0_MM"]

signal_mask = (mass_array > 5230) & (mass_array < 5330)
background_mask = ((mass_array > 5000) & (mass_array < 5150)) | ((mass_array > 5450) & (mass_array < 5700))

training_indices = signal_mask | background_mask

# We define the targets for the whole dataset where applicable
y_all = np.zeros(len(df_clean))
y_all[signal_mask] = 1



# %%
#=================================================================
# We apply k-fold cross-validation by training two separate BDTs on even and odd events to avoid overfitting and to get a more robust performance estimate
#=================================================================

# We train BDT2 (Even events)
print("Training BDT_Even on even events...")
even_train_indices = training_indices & even_mask

X_train_even = df_clean[feature_names].loc[even_train_indices]
y_train_even = y_all[even_train_indices]

bdt_even = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
bdt_even.fit(X_train_even, y_train_even)

# We train BDT1 (Odd events)
print("Training BDT_Odd on odd events...")
odd_train_indices = training_indices & ~even_mask

X_train_odd = df_clean[feature_names].loc[odd_train_indices]
y_train_odd = y_all[odd_train_indices]

bdt_odd = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
bdt_odd.fit(X_train_odd, y_train_odd)

print("Both BDT models trained successfully.")
#%%
#=================================================================
# We evauate the cross-folded scores by applying each BDT to the opposite fold's data
#=================================================================

# We create a new column to store the BDT scores
df_clean["BDT_Score"] = np.nan

# We apply BDT_Even to the odd events 
df_clean.loc[~even_mask, "BDT_Score"] = bdt_even.predict_proba(df_clean.loc[~even_mask, feature_names])[:, 1]

# We apply BDT_Odd to the even events
df_clean.loc[even_mask, "BDT_Score"] = bdt_odd.predict_proba(df_clean.loc[even_mask, feature_names])[:, 1]

#=================================================================
# We calculate an optimal Punzi cut value
#=================================================================

print("Calculating optimal Punzi cut value...")

# 1. We define a range of potential BDT cut values to test
cut_scan_range = np.linspace(0.0, 1.0, 200)

best_punzi_score = -1
bdt_default_cut = None
punzi_scores = []
a = 5.0 # Here we pick a value of a=5 for the Punzi figure of merit

# 2. We isolate the signal and background events based on the mass windows
# We also only look at the data the BDT actually looked at
initial_signal_events = df_clean.loc[signal_mask]
initial_background_events = df_clean.loc[background_mask]
total_initial_signal = len(initial_signal_events)

# 3. We loop over the potential BDT cut values and calculate the Punzi score for each
for cut in cut_scan_range:
    # Count how many pass the cut
    passed_sig = np.sum(initial_signal_events["BDT_Score"] > cut)
    passed_bkg = np.sum(initial_background_events["BDT_Score"] > cut)
    
    # Calculate Signal Efficiency (epsilon_s)
    eff_s = passed_sig / total_initial_signal if total_initial_signal > 0 else 0
    
    # Calculate Punzi's Figure of Merit(FOM)
    denominator = (a / 2.0) + np.sqrt(passed_bkg)
    punzi_fom = eff_s / denominator if denominator > 0 else 0
    punzi_scores.append(punzi_fom)
    
    # Check for maximum
    if punzi_fom > best_punzi_score:
        best_punzi_score = punzi_fom
        bdt_default_cut = cut
    
print(f"Optimal Punzi cut value found: {bdt_default_cut:.4f} with Punzi score: {best_punzi_score:.4f}")


# We apply our calculated optimal Punzi cut selection condition
df_cleaned_by_BDT = df_clean[df_clean["BDT_Score"] > bdt_default_cut]

#%%
#=================================================================
# We define the Mathematical Models (the Shapes)
#=================================================================

# We define the composite model: Gaussian for the signal + Exponential for the background
def fit_model_composite(m, a_sig, mu, sigma, b_bkg, c_bkg):
    signal = a_sig * np.exp(-0.5 * ((m - mu) / sigma)**2)
    # Shifting by 5000 MeV prevents exponential overflow/underflow issues in curve_fit
    background = b_bkg * np.exp(-c_bkg * (m - 5000)) 
    return signal + background

# We define a pure Gaussian model for the purified post-BDT data
def pure_gaussian(m, a_sig, mu, sigma):
    return a_sig * np.exp(-0.5 * ((m - mu) / sigma)**2)

# The plotting parameters
bins = 100
mass_range = (5000, 5700)
m_mesh = np.linspace(5000, 5700, 500)

#=================================================================
#We fit the raw data with the composite model to extract the signal and background shapes
#=================================================================

# We extract the data coordinates (x, y) for the fitting
counts_raw, bin_edges = np.histogram(df_clean["B0_MM"], bins=bins, range=mass_range)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Initial guesses: [amplitude_signal, mean, sigma, amplitude_background, decay_constant]
# B0 mass is roughly 5280 MeV. We guess a standard deviation of ~20 MeV.
initial_guesses_raw = [max(counts_raw), 5280, 20, min(counts_raw), 0.001]

try:
    popt_raw, _ = curve_fit(fit_model_composite, bin_centers, counts_raw, p0=initial_guesses_raw)
    fit_raw_success = True
except Exception as e:
    print(f"Raw data fit failed: {e}")
    fit_raw_success = False

#=================================================================
# We fit the post-BDT data with the pure Gaussian model to check the signal purity
#=================================================================

# We extract the data coordinates (x, y) for the fitting
counts_clean, _ = np.histogram(df_cleaned_by_BDT["B0_MM"], bins=bins, range=mass_range)

# Initial guesses: [amplitude_signal, mean, sigma]
initial_guesses_clean = [max(counts_clean), 5280, 20]

try:
    popt_clean, _ = curve_fit(pure_gaussian, bin_centers, counts_clean, p0=initial_guesses_clean)
    fit_clean_success = True
except Exception as e:
    print(f"Clean data fit failed: {e}")
    fit_clean_success = False

#=================================================================
# We plot before vs after comparison with the fitted curves
#=================================================================
plt.figure(figsize=(11, 6))

# We plot the histograms as steps
plt.hist(df_clean["B0_MM"], bins=bins, range=mass_range, histtype='step', 
         linewidth=1.5, label="Raw data (Pre-selection)", color="red", alpha=0.4)

plt.hist(df_cleaned_by_BDT["B0_MM"], bins=bins, range=mass_range, histtype='step', 
         linewidth=1.5, label=f"Clean data (Punzi BDT score > {bdt_default_cut:.2f})", color="green", alpha=0.4)

# We plot the fitted curves if the fits were successful
if fit_raw_success:
    plt.plot(m_mesh, fit_model_composite(m_mesh, *popt_raw), color="darkred", linestyle="--", linewidth=2.5,
             label=rf"Raw Fit (Gauss+Exp): $\mu$={popt_raw[1]:.1f} MeV, $\sigma$={popt_raw[2]:.1f} MeV")

if fit_clean_success:
    plt.plot(m_mesh, pure_gaussian(m_mesh, *popt_clean), color="darkgreen", linestyle="-", linewidth=2.5,
             label=rf"Clean Fit (Pure Gauss): $\mu$={popt_clean[1]:.1f} MeV, $\sigma$={popt_clean[2]:.1f} MeV")
    
# Plot styling
plt.xlabel('Measured Invariant Mass $B^0_{MM}$ [MeV]')
plt.ylabel('Events/Bin')
plt.title(r'LHCb $B^0 \rightarrow K^{*0}\mu^+\mu^-$ Parametric Mass Fit Comparison')
plt.legend(loc="upper right")
plt.grid(axis='y', alpha=0.3)
plt.show()

# We print the mathematical functions
print("\n" + "="*50)
print("EXTRACTED ANALYTICAL FUNCTIONS")
print("="*50)
if fit_raw_success:
    print(f"Pre-cut Function f(m):\n  {popt_raw[0]:.2f} * exp(-0.5*((m - {popt_raw[1]:.2f})/{popt_raw[2]:.2f})^2) + {popt_raw[3]:.2f} * exp(-{popt_raw[4]:.5f}*(m - 5000))")
if fit_clean_success:
    print(f"\nPost-cut Function f(m):\n  {popt_clean[0]:.2f} * exp(-0.5*((m - {popt_clean[1]:.2f})/{popt_clean[2]:.2f})^2)")
print("="*50)

# %%
#=================================================================
# We verify if there is any overtraining"
#=================================================================
print("Generating BDT Overtraining Validation Plot...")

# Extract scores for BDT_Even
train_scores_even = bdt_even.predict_proba(X_train_even)[:, 1]
test_scores_even = bdt_even.predict_proba(df_clean.loc[odd_train_indices, feature_names])[:, 1]

plt.figure(figsize=(8, 5))

# Plot Signal distributions
plt.hist(train_scores_even[y_train_even == 1], bins=30, range=(0,1), alpha=0.3, 
         color='blue', label='Signal Train (Even)', histtype='step', density=True, linewidth=2)
plt.hist(test_scores_even[y_all[odd_train_indices] == 1], bins=30, range=(0,1), alpha=0.7, 
         color='blue', label='Signal Test (Odd)', histtype='stepfilled', density=True, fill=False, hatch='//')

# Plot Background distributions
plt.hist(train_scores_even[y_train_even == 0], bins=30, range=(0,1), alpha=0.3, 
         color='red', label='Bkg Train (Even)', histtype='step', density=True, linewidth=2)
plt.hist(test_scores_even[y_all[odd_train_indices] == 0], bins=30, range=(0,1), alpha=0.7, 
         color='red', label='Bkg Test (Odd)', histtype='stepfilled', density=True, fill=False, hatch='\\')

plt.legend(loc="upper center")
plt.title("Overtraining Check: BDT_Even Response")
plt.xlabel("BDT Output Score")
plt.ylabel("Arbitrary Units (Normalized)")
plt.yscale('log') # LHCb tip: a log scale makes it much easier to see the background rejection details!
plt.grid(True, alpha=0.2)
plt.show()

print(r"If the Train and Test disctributions shapes match well for both signal and background, it suggests that there is no significant overtraining. Significant discrepancies, especially if the training distribution is much more peaked than the test distribution, could indicate overfitting.")

#%%