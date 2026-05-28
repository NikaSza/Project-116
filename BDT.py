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

# We filter the data based on the following conditions:
cut_conditions = (
    # Particle identification cuts
    (df_raw["Kplus_ProbNNk"] > 0.85) &
    (df_raw["piminus_ProbNNpi"] > 0.20) &
    (df_raw["muplus_isMuon"] != 0) &
    (df_raw["muminus_isMuon"] != 0) &

    # Kinematic cuts
    (df_raw["Kplus_PT"] > 250) &
    (df_raw["piminus_PT"] > 250) &
    (df_raw["muplus_PT"] > 300) &
    (df_raw["muminus_PT"] > 300) &

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

# We apply a BDT cut selection condition 
bdt_cut_value = 0.70
df_cleaned_by_BDT = df_clean[df_clean["BDT_Score"] > bdt_cut_value]

# We plot the Before vs After comparison
plt.figure(figsize=(10, 5))

# Plot before BDT cut
plt.hist(df_clean["B0_MM"], bins=100, range=(5000, 5700), histtype='step', linewidth=2, label="Raw data (Pre-selection)", color="red")
# Plot after BDT cut
plt.hist(df_cleaned_by_BDT["B0_MM"], bins=100, range=(5000, 5700), histtype='step', linewidth=2, label=f"Clean data (BDT score > {bdt_cut_value})", color="green")

plt.xlabel('Measured Invariant Mass $B^0_{MM}$ [MeV]')
plt.ylabel('Events/Bin')
plt.title(r'LHCb $B^0 \rightarrow K^{*0}\mu^+\mu^-$ Invariant Mass Isolation via BDT Sideband Training')
plt.legend(loc="upper right")
plt.grid(axis='y', alpha=0.3)
plt.show()
# %%
