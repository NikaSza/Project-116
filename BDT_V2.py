# %%
import uproot 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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
all_needed = feature_names + ["B0_MM", "Kplus_ProbNNk", "piminus_ProbNNpi", "muplus_isMuon", "muminus_isMuon"] + trigger_branches
df_raw = tree.arrays(all_needed, library="pd")

# We filter the data based on the following conditions: (we removed the particle ID cuts and the kinematic cuts)
cut_conditions = (
    # Trigger cuts
    ((df_raw["B0_L0Global_TOS"] == 1) | (df_raw["B0_Hlt1Phys_TOS"] == 1))
)

# We clean the data by applying the cut conditions
df_clean = df_raw[cut_conditions].copy()

# We define the BDT Mass Windows on our pre-selected data
mass_array = df_clean["B0_MM"]

signal_mask = (mass_array > 5230) & (mass_array < 5330)
background_mask = ((mass_array > 5000) & (mass_array < 5150)) | ((mass_array > 5450) & (mass_array < 5700))

training_indices = signal_mask | background_mask

# We define the target labels for the BDT
X_train_data = df_clean[feature_names].loc[training_indices]
y_train_data = np.zeros(len(df_clean[training_indices]))
y_train_data[signal_mask[training_indices]] = 1

print(f"Training sample prepared. BDT starting now...")


# %%
# We train the BDT model
bdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
bdt.fit(X_train_data, y_train_data)
print("BDT training completed.")


#%%
# We use the BDT to calculate a "signal probability score" for each event in the dataset
df_clean["BDT_Score"] = bdt.predict_proba(df_clean[feature_names])[:, 1]

#=================================================================
# We calculate an optimal Punzi cut value
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
    
    # Calculate Punzi FOM
    denominator = (a / 2.0) + np.sqrt(passed_bkg)
    punzi_fom = eff_s / denominator if denominator > 0 else 0
    punzi_scores.append(punzi_fom)
    
    # Check for maximum
    if punzi_fom > best_punzi_score:
        best_punzi_score = punzi_fom
        bdt_default_cut = cut
    
print(f"Optimal Punzi cut value found: {bdt_default_cut:.4f} with Punzi score: {best_punzi_score:.4f}")
#=================================================================

# We apply our calculated optimal Punzi cut selection condition
df_cleaned_by_BDT = df_clean[df_clean["BDT_Score"] > bdt_default_cut]

# We plot the Before vs After comparison
plt.figure(figsize=(10, 5))

# Plot before BDT cut
plt.hist(df_clean["B0_MM"], bins=100, range=(5000, 5700), histtype='step', linewidth=2, label="Raw data (Pre-selection)", color="red")
# Plot after BDT cut
plt.hist(df_cleaned_by_BDT["B0_MM"], bins=100, range=(5000, 5700), histtype='step', linewidth=2, label=f"Clean data (Punzi BDT score > {bdt_default_cut:.2f})", color="green")

plt.xlabel('Measured Invariant Mass $B^0_{MM}$ [MeV]')
plt.ylabel('Events/Bin')
plt.title(r'LHCb $B^0 \rightarrow K^{*0}\mu^+\mu^-$ Invariant Mass Isolation via BDT Sideband Training')
plt.legend(loc="upper right")
plt.grid(axis='y', alpha=0.3)
plt.show()
# %%
