import uproot 
import pandas as pd
import numpy as np

# We open the ROOT file and access the tree
file = uproot.open(r"C:\Users\Jonas\Documents\Maastricht University\MSP\Year 1\Period 6\PRO116_code\00385270_00000001_1.dvntuple.root")
tree = file["MyDecayTree_muons/DecayTree;1"]

# We read all branches into a pandas DataFrame
all_branches = tree.keys()
df_raw = tree.arrays(all_branches, library="pd")

# We filter the data based on the following conditions:
cut_conditions = (
    (df_raw["Kplus_ProbNNk"] > 0.0) &
    (df_raw["piminus_ProbNNpi"] > 0.0) &
    (df_raw["muplus_isMuon"] != 0) &
    (df_raw["muminus_isMuon"] != 0)
)

# We clean the data by applying the cut conditions
df_clean = df_raw[cut_conditions].copy()

# We define the BDT Mass Windows on our pre-selected data
mass_array = df_clean["B0_MM"]

signal_mask = (mass_array > 5230) & (mass_array < 5330)
background_mask = ((mass_array > 5000) & (mass_array < 5150)) | ((mass_array > 5450) & (mass_array < 5700))

training_indices = signal_mask | background_mask

# We define the target labels for the BDT
y_train = np.zeros(len(df_clean[training_indices]))
y_train[signal_mask[training_indices]] = 1  

# We prepare the features for training the BDT
exclude_vars = ["B0_MM", "B0_ID"]
feature_names = [b for b in all_branches if b not in exclude_vars]

X_train = df_clean[feature_names].loc[training_indices]


print("\n" + "="*40)
print("       BDT DATASET SANITY CHECKS       ")
print("="*40)

# Check 1: Did the pre-selection cuts actually filter rows?
print(f"Original events in ROOT file: {len(df_raw)}")
print(f"Events after teacher's cuts:  {len(df_clean)}")
print(f"Efficiency of cuts:           {len(df_clean)/len(df_raw)*100:.2f}%")
print("-" * 40)

# Check 2: Did we successfully capture signal and background proxy events?
n_signal_proxy = np.sum(y_train == 1)
n_bkg_proxy = np.sum(y_train == 0)
print(f"Total events chosen for BDT training: {len(X_train)}")
print(f"  --> Number of Signal Proxy events (in peak):    {n_signal_proxy}")
print(f"  --> Number of Background Proxy events (sidebands): {n_bkg_proxy}")
print("-" * 40)

# Check 3: Dimension matching (Crucial for XGBoost)
print(f"Shape of feature matrix X_train: {X_train.shape}")
print(f"Length of target vector y_train: {len(y_train)}")

if X_train.shape[0] == len(y_train):
    print("✅ SUCCESS: Row counts match perfectly! Ready for BDT training.")
else:
    print("❌ ERROR: Row count mismatch between X_train and y_train.")
print("="*40 + "\n")